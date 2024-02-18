import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GMM4Bernoulli(nn.Module):
    def __init__(self, pi: torch.FloatTensor, mu: torch.FloatTensor) -> None:
        super().__init__()
        self.pi = nn.Parameter(pi.clone().detach())
        self.mu = nn.Parameter(mu.clone().detach())

    def sample(self, n_samples: int) -> tuple:
        with torch.no_grad():
            weights = torch.tensor([0.5*(1-self.pi), 0.5*(1-self.pi), 0.5*self.pi, 0.5*self.pi])
            zz_list = torch.multinomial(weights, n_samples, replacement=True)
            z_list = torch.normal(self.mu[zz_list], 1.0)
            x_list = torch.bernoulli(torch.sigmoid(z_list))
        return z_list, x_list
    
    def log_pz(self, z_list: torch.FloatTensor) -> torch.Tensor:
        weights = [0.5*(1-self.pi), 0.5*(1-self.pi), 0.5*self.pi, 0.5*self.pi]
        components_log_likelihood = torch.zeros((4, z_list.shape[0]))
        for i in range(4):
            components_log_likelihood[i] = weights[i].log() - F.gaussian_nll_loss(self.mu[i], z_list, torch.tensor(1.), full=True, reduction='none')
        ln_pz_list = torch.logsumexp(components_log_likelihood, dim=0)
        return ln_pz_list

    def log_pxgz(self, z_list: torch.FloatTensor, x_list: torch.FloatTensor) -> torch.Tensor:
        p = torch.sigmoid(z_list)
        if len(x_list.shape) == 0:
            x_list = x_list.expand(z_list.shape[0])
        ln_pxgz_list = -F.binary_cross_entropy(p, x_list, reduction='none')
        return ln_pxgz_list

    def complete_log_likelihood(self, z_list: torch.FloatTensor, x_list: torch.FloatTensor) -> torch.Tensor:
        return self.log_pz(z_list) + self.log_pxgz(z_list, x_list)

    
class Post2Gaussian(nn.Module):
    def __init__(self, mu: torch.FloatTensor, sigma: torch.FloatTensor) -> None:
        super().__init__()
        self.mu = nn.Parameter(mu.clone().detach())
        self.sigma = nn.Parameter(sigma.clone().detach())
    
    def sample(self, x: torch.FloatTensor, n_samples: int):
        # with torch.no_grad():
        #     if x == 0:
        #         z_list = torch.normal(self.mu[0].item(), self.sigma[0].item(), size=(n_samples,))
        #     else:
        #         z_list = torch.normal(self.mu[1].item(), self.sigma[1].item(), size=(n_samples,))
        z_list = torch.normal(0, 1, size=(n_samples,)) * self.sigma[int(x)].abs() + self.mu[int(x)]
        return z_list
    
    def hid_log_likelihood(self, z_list: torch.FloatTensor, x: torch.FloatTensor) -> torch.FloatTensor:
        if x == 0:
            return -F.gaussian_nll_loss(self.mu[0], z_list, self.sigma[0]**2, full=True, reduction='none')
        else:
            return -F.gaussian_nll_loss(self.mu[1], z_list, self.sigma[1]**2, full=True, reduction='none')