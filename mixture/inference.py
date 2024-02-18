import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def elbo(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor, grad: str) -> torch.FloatTensor:
    """ELBO with score gradients well prepared. Only for score.
    """
    
    if grad == 'score':
        ln_p_list_values = ln_p_list.detach().clone()
        ln_q_list_values = ln_q_list.detach().clone()
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
        return 1/2 * result + 1/2 * result.detach().clone()
    elif grad == 'pathwise':
        return result
    

# def dFKLpq(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor, grad: str = None) -> torch.FloatTensor:
#     """Optimize the variational distribution by minimizing the forward KL divergence. For both score and pathwise.
#     """
#     return torch.logsumexp(ln_p_list - ln_q_list, dim=0) - np.log(ln_q_list.shape[0])


def dFKLpq(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor, grad: str = None) -> torch.FloatTensor:
    """Optimize the variational distribution by minimizing the forward KL divergence. For both score and pathwise.
    """

    if grad == 'score':
        return (torch.logsumexp(ln_p_list - ln_q_list, dim=0) - np.log(ln_q_list.shape[0])).exp()
    elif grad == 'pathwise':
        temp = ln_p_list - ln_q_list
        return (temp.exp() * temp).mean()
        # return (temp.exp() * (-ln_q_list)).mean()


def variational_importance_sampling(
        inf_model,
        vari_model,
        inf_optimizer: torch.optim.Optimizer,
        vari_optimizer: torch.optim.Optimizer,
        x_list: torch.FloatTensor,
        n_epochs: int = 1000,
        n_monte_carlo: int = 1000,
        batch_size: int = 20,
        grad: str = 'score',
        update: str = 'ELBO',
        divergence: str = 'chi^2',
        print_freq: int = 100) -> torch.FloatTensor:
    
    n_samples = len(x_list)
    n_batches = int(n_samples / batch_size)
    
    epoch_loss_list = torch.zeros(2, n_epochs)

    record_mu = np.zeros((4, n_epochs + 1))
    record_pi = np.zeros(n_epochs + 1)
    record_mu[:, 0] = inf_model.mu.detach().numpy()
    record_pi[0] = inf_model.pi.item()

    for epoch in range(n_epochs):
        for batch in range(n_batches):
            batch_loss = 0
            z_list_list = [None for i in range(batch_size)]
            ln_q_list_list = torch.zeros((batch_size, n_monte_carlo))

            for sample in range(batch*batch_size, (batch+1)*batch_size):
                x = x_list[sample]
                
                # prepare sample for updating q
                if grad == 'score':
                    with torch.no_grad():
                        z_list = vari_model.sample(x, n_samples=n_monte_carlo)
                elif grad == 'pathwise':
                    z_list = vari_model.sample(x, n_samples=n_monte_carlo)
                ln_q_list = vari_model.hid_log_likelihood(z_list, x)
                z_list_list[sample - batch*batch_size] = z_list
                ln_q_list_list[sample - batch*batch_size] = ln_q_list
                
                # update p
                ln_p_list = inf_model.complete_log_likelihood(z_list.detach(), x)
                if update == 'ELBO':
                    batch_loss -= (ln_p_list - ln_q_list.detach()).mean()
                elif update == 'marginal':
                    batch_loss -= marginal_log_likelihood(ln_p_list, ln_q_list.detach())
                    
            batch_loss /= batch_size
            batch_loss.backward()
            inf_optimizer.step()
            inf_optimizer.zero_grad()

            with torch.no_grad():
                inf_model.pi.clamp_(min=0.01, max=0.99)
            
            epoch_loss_list[0, epoch] += batch_loss.item()

            # update q
            batch_loss = 0
            for sample in range(batch*batch_size, (batch+1)*batch_size):
                i = sample - batch*batch_size

                ln_p_list = inf_model.complete_log_likelihood(z_list_list[i], x_list[sample])
                if divergence == 'RKL':
                    batch_loss -= elbo(ln_p_list, ln_q_list_list[i], grad)
                elif divergence == 'FKL':
                    batch_loss += dFKLpq(ln_p_list, ln_q_list_list[i], grad)
                elif divergence == 'chi^2':
                    batch_loss += log_V(ln_p_list, ln_q_list_list[i], grad)
                elif divergence == 'sandwich':
                    batch_loss += -0.5 * elbo(ln_p_list, ln_q_list_list[i], grad) + 0.5 * log_V(ln_p_list, ln_q_list_list[i], grad)
            
            batch_loss /= batch_size
            batch_loss.backward()
            vari_optimizer.step()
            inf_optimizer.zero_grad()
            vari_optimizer.zero_grad()

            epoch_loss_list[1, epoch] += batch_loss.item()
            
        epoch_loss_list[:, epoch] /= n_batches
        if epoch % print_freq == 0:
            print(epoch, epoch_loss_list[:, epoch], flush=True)
        record_mu[:, epoch + 1] = inf_model.mu.detach().numpy()
        record_pi[epoch + 1] = inf_model.pi.item()
    return epoch_loss_list, record_mu, record_pi



def evaluate(inf_model, vari_model, x_list, z_list, n_monte_carlo: int = 1000, seed: int = 0):
    n_samples = x_list.shape[0]
    
    df = pd.DataFrame(index=np.arange(n_samples), columns=['pred complete log-likelihood', 'hidden log-likelihood', 'marginal log-likelihood', 'ELBO', 'CUBO', 'marginal log-likelihood uniform', 'sample'])
    
    torch.manual_seed(seed)

    with torch.no_grad():
        for sample in range(n_samples):
            x = x_list[sample]
            df.at[sample, 'pred complete log-likelihood'] = inf_model.complete_log_likelihood(z_list[sample:sample+1], x)[0].item()
            df.at[sample, 'hidden log-likelihood'] = vari_model.hid_log_likelihood(z_list[sample:sample+1], x)[0].item()

            z_list_sampled = vari_model.sample(x, n_samples=n_monte_carlo)
            ln_p_list = inf_model.complete_log_likelihood(z_list_sampled, x)
            ln_q_list = vari_model.hid_log_likelihood(z_list_sampled, x)
            df.at[sample, 'marginal log-likelihood'] = marginal_log_likelihood(ln_p_list, ln_q_list).item()
            df.at[sample, 'ELBO'] = (ln_p_list - ln_q_list).mean().item()
            df.at[sample, 'CUBO'] = 0.5 * log_V(ln_p_list, ln_q_list, 'pathwise').item()

            z_list_sampled = torch.linspace(-15, 15, n_monte_carlo)
            ln_p_list = inf_model.complete_log_likelihood(z_list_sampled, x)
            ln_q_list = torch.log(1/30 * torch.ones(n_monte_carlo))
            df.at[sample, 'marginal log-likelihood uniform'] = (marginal_log_likelihood(ln_p_list, ln_q_list).item())

            df.at[sample, 'sample'] = sample
    return df