import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# def elbo(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor, grad: str) -> torch.FloatTensor:
#     """ELBO with score gradients well prepared. Only for score.
#     """
    
#     if grad == 'score':
#         ln_p_list_values = ln_p_list.detach().clone()
#         ln_q_list_values = ln_q_list.detach().clone()
#         elbo_values = ln_p_list_values - ln_q_list_values
#         return torch.mean(ln_p_list - ln_p_list_values + elbo_values * (ln_q_list - ln_q_list_values) + elbo_values)
#     elif grad == 'pathwise':
#         return (ln_p_list - ln_q_list).mean()
#     else:
#         raise ValueError('grad parameter is not in [score | pathwise].')

def elbo(ln_pxgz_list: torch.FloatTensor, ln_q_list: torch.FloatTensor, grad: str, kl_term) -> torch.FloatTensor:
    if grad == 'score':
        return (ln_pxgz_list.detach() * ln_q_list).mean() - kl_term
    elif grad == 'pathwise':
        return ln_pxgz_list.mean() - kl_term


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
    

# def dFKLpq(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor, grad: str = None) -> torch.FloatTensor:
#     """Optimize the variational distribution by minimizing the forward KL divergence. For both score and pathwise.
#     """

#     if grad == 'score':
#         return (torch.logsumexp(ln_p_list - ln_q_list, dim=0) - np.log(ln_q_list.shape[0])).exp()
#     elif grad == 'pathwise':
#         temp = ln_p_list - ln_q_list
#         return (temp.exp() * temp).mean()
#         # return (temp.exp() * (-ln_q_list)).mean()


def variational_importance_sampling(
        inf_model,
        vari_model,
        inf_optimizer: torch.optim.Optimizer,
        vari_optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        test_data,
        method: str,
        seed: int,
        n_epochs: int = 1000,
        n_monte_carlo: int = 1000,
        grad: str = 'score',
        update: str = 'ELBO',
        divergence: str = 'chi^2',
        print_freq: int = 100,) -> torch.FloatTensor:
    
    epoch_loss_list = torch.zeros(2, n_epochs)

    for epoch in range(n_epochs):
        for x_list, __ in dataloader:
            batch_size = x_list.shape[0]
            loss = 0

            z_list_list = [None for i in range(batch_size)]
            ln_q_list_list = [None for i in range(batch_size)]
            kl_list = [None for i in range(batch_size)]

            for sample in range(batch_size):
                x = x_list[sample].reshape((-1))
                mu, log_sigma = vari_model.forward(x)
                if grad == 'score':
                    with torch.no_grad():
                        z_list = vari_model.sample(mu=mu, log_sigma=log_sigma, n_samples=n_monte_carlo)
                elif grad == 'pathwise':
                    z_list = vari_model.sample(mu=mu, log_sigma=log_sigma, n_samples=n_monte_carlo)

                ln_q_list = vari_model.hid_log_likelihood(z_list, mu=mu, log_sigma=log_sigma)
                ln_pz_list = inf_model.prior_log_likelihood(z_list.detach())
                ln_pxgz_list = inf_model.conditional_log_likelihood(z_list.detach(), x)
                kl_term = vari_model.kl(mu=mu, log_sigma=log_sigma)

                z_list_list[sample] = z_list
                ln_q_list_list[sample] = ln_q_list
                kl_list[sample] = kl_term
        
                if update == 'ELBO':
                    loss -= elbo(ln_pxgz_list, ln_q_list.detach(), 'pathwise', kl_term.detach())
                elif update == 'marginal':
                    ln_p_list = ln_pz_list + ln_pxgz_list
                    loss -= marginal_log_likelihood(ln_p_list, ln_q_list.detach())
            
            loss /= batch_size
            loss.backward()
            inf_optimizer.step()
            inf_optimizer.zero_grad()
            vari_optimizer.zero_grad()

            print(loss.item(), flush=True)
            
            epoch_loss_list[0, epoch] += loss.item()

            # update q
            loss = 0
            for sample in range(batch_size):
                x = x_list[sample].reshape((-1))
                z_list = z_list_list[sample]
                ln_q_list = ln_q_list_list[sample]
                kl_term = kl_list[sample]

                ln_pz_list = inf_model.prior_log_likelihood(z_list)
                ln_pxgz_list = inf_model.conditional_log_likelihood(z_list, x)
            
                if divergence == 'RKL':
                    loss -= elbo(ln_pxgz_list, ln_q_list, grad, kl_term)
                # elif divergence == 'FKL':
                #     ln_p_list = ln_pz_list + ln_pxgz_list
                #     loss = dFKLpq(ln_p_list, ln_q_list, grad)
                elif divergence == 'chi^2':
                    ln_p_list = ln_pz_list + ln_pxgz_list
                    loss += log_V(ln_p_list, ln_q_list, grad)
                elif divergence == 'sandwich':
                    ln_p_list = ln_pz_list + ln_pxgz_list
                    loss += -0.5 * elbo(ln_pxgz_list, ln_q_list, grad, kl_term) + 0.5 * log_V(ln_p_list, ln_q_list, grad)
                elif divergence == 'importance':
                    if grad == 'score':
                        loss -= torch.tensor(0.)
                    elif grad == 'pathwise':
                        ln_p_list = ln_pz_list + ln_pxgz_list
                        loss -= marginal_log_likelihood(ln_p_list, ln_q_list)
            
            loss /= batch_size
            loss.backward()
            vari_optimizer.step()
            inf_optimizer.zero_grad()
            vari_optimizer.zero_grad()

            epoch_loss_list[1, epoch] += loss.item()
        epoch_loss_list[:, epoch] /= len(dataloader)
            
        if epoch % print_freq == 0:
            print(epoch, epoch_loss_list[:, epoch], flush=True)
        
        torch.save(inf_model.state_dict(), f'model/{method}_{seed}_{epoch}_inf.pt')
        torch.save(vari_model.state_dict(), f'model/{method}_{seed}_{epoch}_vari.pt')

        ## evaluate
        df = evaluate(inf_model, vari_model, test_data, 1000).mean().to_frame().T
        df.to_csv(f'csv/{method}_{seed}_{epoch}.csv', index=False)
    return epoch_loss_list



def evaluate(inf_model, vari_model, dataset, n_monte_carlo: int = 1000, seed: int = 0):
    n_samples = len(dataset)
    df = pd.DataFrame(index=np.arange(n_samples), columns=['marginal log-likelihood', 'ELBO', 'conditional log-likelihood'])
    
    torch.manual_seed(seed)
    
    with torch.no_grad():
        for sample in range(n_samples):
            x = dataset[sample][0].reshape((-1))
            mu, log_sigma = vari_model.forward(x)
            z_list = vari_model.sample(mu=mu, log_sigma=log_sigma, n_samples=n_monte_carlo)
            ln_q_list = vari_model.hid_log_likelihood(z_list, mu=mu, log_sigma=log_sigma)
            ln_pz_list = inf_model.prior_log_likelihood(z_list)
            ln_pxgz_list = inf_model.conditional_log_likelihood(z_list, x)
            kl_term = vari_model.kl(mu=mu, log_sigma=log_sigma)
            df.at[sample, 'marginal log-likelihood'] = marginal_log_likelihood(ln_pz_list + ln_pxgz_list, ln_q_list).item()
            df.at[sample, 'ELBO'] = (ln_pxgz_list - kl_term).mean().item()
            df.at[sample, 'conditional log-likelihood'] = inf_model.conditional_log_likelihood(mu.expand((1, -1)), x).item()
    return df


def evaluate2(inf_model, dataset, n_monte_carlo: int = 1000, seed: int = 0):
    n_samples = len(dataset)
    df = pd.DataFrame(index=np.arange(n_samples), columns=['marginal log-likelihood'])
    
    torch.manual_seed(seed)
    
    with torch.no_grad():
        for sample in range(n_samples):
            x = dataset[sample][0].reshape((-1))
            z_list = torch.randn((n_monte_carlo, 2)) * 1.3
            ln_q_list = -F.gaussian_nll_loss(torch.zeros((n_monte_carlo, 2)), z_list, torch.tensor(1.3)**2 * torch.ones((n_monte_carlo, 2)), full=True, reduction='none').sum(dim=1)
            ln_pz_list = inf_model.prior_log_likelihood(z_list)
            ln_pxgz_list = inf_model.conditional_log_likelihood(z_list, x)
            df.at[sample, 'marginal log-likelihood'] = marginal_log_likelihood(ln_pz_list + ln_pxgz_list, ln_q_list).item()
    return df