import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def exp_basis(decay: float, window_size: int, time_span: float):
    basis = torch.zeros(window_size)
    dt = time_span / window_size
    t = torch.linspace(dt, time_span, window_size)
    basis = torch.exp(-decay * t)
    basis /= (dt * basis.sum(axis=0)) # normalization
    return basis


def poisson_log_likelihood(lam: torch.FloatTensor, k: torch.FloatTensor) -> torch.FloatTensor:
    return k * (lam+1e-8).log() - lam - torch.lgamma(k+1)


def convolve_spikes_with_basis(spikes_list: torch.FloatTensor, basis: torch.FloatTensor) -> torch.FloatTensor:
    window_size = len(basis)
    n_seq, n_time_bins, n_neurons = spikes_list.shape
    padded_spikes_list = torch.cat((torch.zeros((n_seq, window_size, n_neurons)), spikes_list), dim=-2)
    convolved_spikes_list = torch.zeros_like(spikes_list)
    for i in range(window_size):
        convolved_spikes_list = convolved_spikes_list + basis[-(i+1)] * padded_spikes_list[:, i:n_time_bins+i, :]
    return convolved_spikes_list