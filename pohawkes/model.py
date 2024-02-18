import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class POGLM(nn.Module):
    def __init__(self, n_neurons: int, n_vis_neurons: int, basis: torch.FloatTensor) -> None:
        super().__init__()

        self.n_neurons = n_neurons
        self.n_vis_neurons = n_vis_neurons
        self.n_hid_neurons = n_neurons - n_vis_neurons
        
        self.basis = basis
        self.flipped_basis = torch.flip(self.basis, (0,))
        self.window_size = len(self.basis)

        self.linear = nn.Linear(n_neurons, n_neurons)

    def forward(self, convolved_spikes_list: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sigmoid(self.linear(convolved_spikes_list))

    def complete_log_likelihood(
            self,
            hid_spikes_list: torch.FloatTensor,
            convolved_hid_spikes_list: torch.FloatTensor,
            vis_spikes: torch.FloatTensor,
            convolved_vis_spikes: torch.FloatTensor
            ) -> torch.FloatTensor:
        n_samples = convolved_hid_spikes_list.shape[0]
        convolved_spikes_list = torch.cat((convolved_vis_spikes.expand((n_samples, -1, -1)), convolved_hid_spikes_list), dim=2)
        spikes_list = torch.cat((vis_spikes.expand((n_samples, -1, -1)), hid_spikes_list), dim=2)
        firing_rates_list = self.forward(convolved_spikes_list)
        return utils.poisson_log_likelihood(firing_rates_list, spikes_list).sum(dim=(1, 2))
    
    def sample(self, n_time_bins: int, n_samples: int = 1) -> torch.FloatTensor:
        with torch.no_grad():
            spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_neurons))
            convolved_spikes_list = torch.zeros((n_samples, n_time_bins, self.n_neurons))
            firing_rates_list = torch.zeros((n_samples, n_time_bins, self.n_neurons))

            for t in range(n_time_bins):
                convolved_spikes_list[:, t, :] = self.flipped_basis @ spikes_list[:, t:t+self.window_size, :]
                firing_rates_list[:, t, :] = self.forward(convolved_spikes_list[:, t, :])
                spikes_list[:, t+self.window_size, :] = torch.poisson(firing_rates_list[:, t, :])
            spikes_list = spikes_list[:, self.window_size:, :]
            return spikes_list, convolved_spikes_list, firing_rates_list
    

class ForwardSelf(nn.Module):
    def __init__(self, n_neurons: int, n_vis_neurons: int, basis: torch.FloatTensor) -> None:
        super().__init__()

        self.n_neurons = n_neurons
        self.n_vis_neurons = n_vis_neurons
        self.n_hid_neurons = n_neurons - n_vis_neurons
        
        self.basis = basis
        self.flipped_basis = torch.flip(self.basis, (0,))
        self.window_size = len(self.basis)

        self.linear = nn.Linear(n_neurons, self.n_hid_neurons)
    
    def forward(self, convolved_spikes_list: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sigmoid(self.linear(convolved_spikes_list))
    
    def hid_log_likelihood(self, hid_spikes_list: torch.FloatTensor, convolved_hid_spikes_list: torch.FloatTensor = None, convolved_vis_spikes: torch.FloatTensor = None, hid_firing_rates_list: torch.FloatTensor = None) -> torch.FloatTensor:
        n_samples = hid_spikes_list.shape[0]
        if hid_firing_rates_list is None:
            hid_firing_rates_list = self.forward(torch.cat((convolved_vis_spikes.expand((n_samples, -1, -1)), convolved_hid_spikes_list), dim=2))
        return utils.poisson_log_likelihood(hid_firing_rates_list, hid_spikes_list).sum(dim=(1, 2))

    def sample(self, convolved_vis_spikes: torch.FloatTensor, n_samples: int = 1) -> torch.FloatTensor:
        n_time_bins = convolved_vis_spikes.shape[0]
        hid_spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_hid_neurons))
        convolved_hid_spikes_list = torch.zeros((n_samples, n_time_bins, self.n_hid_neurons))
        hid_firing_rates_list = torch.zeros((n_samples, n_time_bins, self.n_hid_neurons))
        
        for t in range(n_time_bins):
            convolved_hid_spikes_list[:, t, :] = self.flipped_basis @ hid_spikes_list[:, t:t+self.window_size, :]
            hid_firing_rates_list[:, t, :] = self.forward(torch.cat((
                convolved_vis_spikes[t, :].expand((n_samples, -1)),
                convolved_hid_spikes_list[:, t, :]
            ), dim=1))
            with torch.no_grad():
                hid_spikes_list[:, t+self.window_size, :] = torch.poisson(hid_firing_rates_list[:, t, :])
        hid_spikes_list = hid_spikes_list[:, self.window_size:, :]
        return hid_spikes_list, convolved_hid_spikes_list, hid_firing_rates_list