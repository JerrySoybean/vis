import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def elbo(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor, grad: str) -> torch.FloatTensor:
    """ELBO with score gradients well prepared. Only for score.
    """
    
    if grad == 'score':
        ln_p_list_values = ln_p_list.detach()
        ln_q_list_values = ln_q_list.detach()
        elbo_values = ln_p_list_values - ln_q_list_values
        return torch.mean(ln_p_list - ln_p_list_values + elbo_values * (ln_q_list - ln_q_list_values) + elbo_values)
    elif grad == 'pathwise':
        return (ln_p_list - ln_q_list).mean()
    else:
        raise ValueError('grad parameter is not in [score | pathwise].')


def marginal_log_likelihood(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor) -> torch.FloatTensor:
    return torch.logsumexp(ln_p_list - ln_q_list, dim=0) - np.log(ln_q_list.shape[0])


def log_V(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor, grad: str) -> torch.FloatTensor:
    """Optimize the variational distribution by minimizing the forward chi^2 divergence. For both score and pathwise.
    """
    
    result = torch.logsumexp(2*(ln_p_list - ln_q_list), dim=0) - np.log(ln_q_list.shape[0])
    if grad == 'score':
        return 1/2 * result + 1/2 * result.detach()
    elif grad == 'pathwise':
        return result


def variational_importance_sampling(
        inf_model,
        vari_model,
        inf_optimizer: torch.optim.Optimizer,
        vari_optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        n_epochs: int = 1000,
        n_monte_carlo: int = 1000,
        update: str = 'ELBO',
        divergence: str = 'chi^2',
        print_freq: int = 100) -> torch.FloatTensor:
    
    epoch_loss_list = torch.zeros(2, n_epochs)

    for epoch in range(n_epochs):
        for vis_spikes_list, convolved_vis_spikes_list in dataloader:
            batch_size = vis_spikes_list.shape[0]
            loss = 0

            hid_spikes_list_list = [None for i in range(batch_size)]
            convolved_hid_spikes_list_list = [None for i in range(batch_size)]
            ln_q_list_list = [None for i in range(batch_size)]

            for sample in range(batch_size):
                vis_spikes = vis_spikes_list[sample]
                convolved_vis_spikes = convolved_vis_spikes_list[sample]
                hid_spikes_list, convolved_hid_spikes_list, hid_firing_rates_list = vari_model.sample(convolved_vis_spikes, n_samples=n_monte_carlo)

                ln_q_list = vari_model.hid_log_likelihood(hid_spikes_list, hid_firing_rates_list=hid_firing_rates_list)
                ln_p_list = inf_model.complete_log_likelihood(hid_spikes_list, convolved_hid_spikes_list, vis_spikes, convolved_vis_spikes)

                hid_spikes_list_list[sample] = hid_spikes_list
                convolved_hid_spikes_list_list[sample] = convolved_hid_spikes_list
                ln_q_list_list[sample] = ln_q_list
        
                if update == 'ELBO':
                    loss -= elbo(ln_p_list, ln_q_list.detach(), 'pathwise')
                elif update == 'marginal':
                    loss -= marginal_log_likelihood(ln_p_list, ln_q_list.detach())
            
            loss /= batch_size
            loss.backward()
            inf_optimizer.step()
            inf_optimizer.zero_grad()
            vari_optimizer.zero_grad()
            
            epoch_loss_list[0, epoch] += loss.item()

            # update q
            loss = 0
            for sample in range(batch_size):
                vis_spikes = vis_spikes_list[sample]
                convolved_vis_spikes = convolved_vis_spikes_list[sample]
                hid_spikes_list = hid_spikes_list_list[sample]
                convolved_hid_spikes_list = convolved_hid_spikes_list_list[sample]
                ln_q_list = ln_q_list_list[sample]

                with torch.no_grad():
                    ln_p_list = inf_model.complete_log_likelihood(hid_spikes_list, convolved_hid_spikes_list, vis_spikes, convolved_vis_spikes)
            
                if divergence == 'RKL':
                    loss -= elbo(ln_p_list, ln_q_list, 'score')
                elif divergence == 'chi^2':
                    loss += log_V(ln_p_list, ln_q_list, 'score')
                elif divergence == 'sandwich':
                    loss += -0.5 * elbo(ln_p_list, ln_q_list, 'score') + 0.5 * log_V(ln_p_list, ln_q_list, 'score')
            
            loss /= batch_size
            loss.backward()
            vari_optimizer.step()
            inf_optimizer.zero_grad()
            vari_optimizer.zero_grad()

            epoch_loss_list[1, epoch] += loss.item()
        epoch_loss_list[:, epoch] /= len(dataloader)
            
        if epoch % print_freq == 0:
            print(epoch, epoch_loss_list[:, epoch], flush=True)
    return epoch_loss_list



def evaluate(inf_model, vari_model, spikes_list, convolved_spikes_list, n_monte_carlo: int = 1000, seed: int = 0):
    n_samples = spikes_list.shape[0]
    df = pd.DataFrame(index=np.arange(n_samples), columns=['pred complete log-likelihood', 'marginal log-likelihood', 'ELBO', 'hid log-likelihood'])
    
    torch.manual_seed(seed)
    
    with torch.no_grad():
        for sample in range(n_samples):
            spikes = spikes_list[sample]
            convolved_spikes = convolved_spikes_list[sample]
            vis_spikes = spikes[:, :inf_model.n_vis_neurons]
            convolved_vis_spikes = convolved_spikes[:, :inf_model.n_vis_neurons]
            hid_spikes_list = spikes[None, :, inf_model.n_vis_neurons:]
            convolved_hid_spikes_list = convolved_spikes[None, :, inf_model.n_vis_neurons:]
            
            df.at[sample, 'pred complete log-likelihood'] = inf_model.complete_log_likelihood(hid_spikes_list, convolved_hid_spikes_list, vis_spikes, convolved_vis_spikes)[0].item()
            df.at[sample, 'hid log-likelihood'] = vari_model.hid_log_likelihood(hid_spikes_list, convolved_hid_spikes_list, convolved_vis_spikes)[0].item()

            hid_spikes_list, convolved_hid_spikes_list, hid_firing_rates_list = vari_model.sample(convolved_vis_spikes, n_samples=n_monte_carlo)
            ln_q_list = vari_model.hid_log_likelihood(hid_spikes_list, hid_firing_rates_list=hid_firing_rates_list)
            ln_p_list = inf_model.complete_log_likelihood(hid_spikes_list, convolved_hid_spikes_list, vis_spikes, convolved_vis_spikes)
            df.at[sample, 'marginal log-likelihood'] = marginal_log_likelihood(ln_p_list, ln_q_list).item()
            df.at[sample, 'ELBO'] = (ln_p_list - ln_q_list).mean().item()
    return df


def evaluate_rgc(inf_model, vari_model, vis_spikes_list, convolved_vis_spikes_list, n_monte_carlo: int = 1000, seed: int = 0):
    n_samples = vis_spikes_list.shape[0]
    df = pd.DataFrame(index=np.arange(n_samples), columns=['marginal log-likelihood', 'ELBO'])
    
    torch.manual_seed(seed)
    
    with torch.no_grad():
        for sample in range(n_samples):
            vis_spikes = vis_spikes_list[sample]
            convolved_vis_spikes = convolved_vis_spikes_list[sample]

            hid_spikes_list, convolved_hid_spikes_list, hid_firing_rates_list = vari_model.sample(convolved_vis_spikes, n_samples=n_monte_carlo)
            ln_q_list = vari_model.hid_log_likelihood(hid_spikes_list, hid_firing_rates_list=hid_firing_rates_list)
            ln_p_list = inf_model.complete_log_likelihood(hid_spikes_list, convolved_hid_spikes_list, vis_spikes, convolved_vis_spikes)
            df.at[sample, 'marginal log-likelihood'] = marginal_log_likelihood(ln_p_list, ln_q_list).item()
            df.at[sample, 'ELBO'] = (ln_p_list - ln_q_list).mean().item()
    return df