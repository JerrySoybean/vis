import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, z_list):
        return torch.sigmoid(self.fc2(torch.tanh((self.fc1(z_list)))))

    def prior_log_likelihood(self, z_list: torch.FloatTensor) -> torch.FloatTensor:
        n_samples = z_list.shape[0]
        ln_pz_list = -F.gaussian_nll_loss(torch.zeros((n_samples, 2)), z_list, torch.ones((n_samples, 2)), full=True, reduction='none').sum(dim=1)
        return ln_pz_list
    
    def conditional_log_likelihood(self, z_list: torch.FloatTensor, x: torch.FloatTensor) -> torch.FloatTensor:
        n_samples = z_list.shape[0]
        x_pred_list = self.forward(z_list)
        ln_pxgz_list = -F.binary_cross_entropy(x_pred_list, x.expand((n_samples, -1)), reduction='none').sum(dim=1)
        return ln_pxgz_list
    
    def complete_log_likelihood(self, z_list: torch.FloatTensor, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.prior_log_likelihood(z_list) + self.conditional_log_likelihood(z_list, x)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(784, 128)
        self.fc_mu = nn.Linear(128, 2)
        self.fc_log_sigma = nn.Linear(128, 2)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        h = torch.tanh(self.fc(x))
        return self.fc_mu(h), self.fc_log_sigma(h)

    def sample(self, x: torch.FloatTensor = None, mu: torch.FloatTensor = None, log_sigma: torch.FloatTensor = None, n_samples: int = 1) -> torch.FloatTensor:
        if x is not None:
            mu, log_sigma = self.forward(x)
        z_list = torch.randn((n_samples, 2), device=mu.device) * log_sigma.exp() + mu
        return z_list

    def hid_log_likelihood(self, z_list: torch.FloatTensor, x: torch.FloatTensor = None, mu: torch.FloatTensor = None, log_sigma: torch.FloatTensor = None) -> torch.FloatTensor:
        n_samples = z_list.shape[0]
        if x is not None:
            mu, log_sigma = self.forward(x)
        ln_qz_list = -F.gaussian_nll_loss(mu.expand((n_samples, -1)), z_list, (log_sigma.exp()**2).expand((n_samples, -1)), full=True, reduction='none').sum(dim=1)
        return ln_qz_list

    def kl(self, x: torch.FloatTensor = None, mu: torch.FloatTensor = None, log_sigma: torch.FloatTensor = None) -> torch.FloatTensor:
        if x is not None:
            mu, log_sigma = self.forward(x)
        return (-log_sigma + (mu**2 + log_sigma.exp()**2 - 1) / 2).sum()
