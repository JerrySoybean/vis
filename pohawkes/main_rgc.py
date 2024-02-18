import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.stats import binned_statistic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import time
import argparse

import model, inference, utils


## arguments
parser = argparse.ArgumentParser()
parser.add_argument('idx', type=int)
args = parser.parse_args()

method_list = ['VI', 'CHIVI', 'VBIS', 'VIS']
n_hid_neurons_list = [1, 2, 3]
seed_list = np.arange(10)

arg_index = np.unravel_index(args.idx, (len(method_list), len(n_hid_neurons_list), len(seed_list)))
method, n_hid_neurons, seed = method_list[arg_index[0]], n_hid_neurons_list[arg_index[1]], seed_list[arg_index[2]]


def load_data():
    temp = loadmat(f'SpTimesRGC.mat', squeeze_me=False, struct_as_record=False)['SpTimes'][0]
    n_time_bins = 20 * 60 * 120 # 20 min * 119.9820 Hz
    time_bins = np.linspace(1, n_time_bins, n_time_bins)
    n_neurons = 27
    spikes = np.zeros((n_time_bins, n_neurons))
    for i in range(n_neurons):
        spikes[:, i] = binned_statistic(temp[i][:, 0], None, bins=np.hstack(([0], time_bins)), statistic='count')[0].T
    return spikes

spikes = load_data()

## hyper-parameters
decay = 5
# dt = 1/120
dt = 0.05
window_size = 1
n_vis_neurons = spikes.shape[1]
n_neurons = n_vis_neurons + n_hid_neurons
basis = utils.exp_basis(decay, window_size, dt*window_size)


vis_spikes_list_train, vis_spikes_list_test = torch.from_numpy(spikes[:96000].reshape(960, 100, -1)).to(torch.float32), torch.from_numpy(spikes[96000:].reshape(480, 100, -1)).to(torch.float32)
convolved_vis_spikes_list_train = utils.convolve_spikes_with_basis(vis_spikes_list_train, basis)
convolved_vis_spikes_list_test = utils.convolve_spikes_with_basis(vis_spikes_list_test, basis)
train_dataset = TensorDataset(vis_spikes_list_train, convolved_vis_spikes_list_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)


n_epochs = 10
print_freq = 1
n_monte_carlo = {1: 1000, 2: 2000, 3: 3000}[n_hid_neurons]

torch.manual_seed(seed)
inf_model = model.POGLM(n_neurons, n_vis_neurons, basis)
vari_model = model.ForwardSelf(n_neurons, n_vis_neurons, basis)
with torch.no_grad():
    inf_model.linear.weight.data = torch.zeros((n_neurons, n_neurons))
    inf_model.linear.bias.data = torch.zeros((n_neurons, ))
    vari_model.linear.weight.data = torch.zeros((n_neurons - n_vis_neurons, n_neurons))
    vari_model.linear.bias.data = torch.zeros((n_neurons - n_vis_neurons, ))

inf_optimizer = torch.optim.Adam(inf_model.parameters(), lr=0.01)
vari_optimizer = torch.optim.Adam(vari_model.parameters(), lr=0.01)


if method == 'VI':
    update, divergence = 'ELBO', 'RKL'
elif method == 'CHIVI':
    update, divergence = 'ELBO', 'sandwich'
elif method == 'VBIS':
    update, divergence = 'marginal', 'RKL'
elif method == 'VIS':
    update, divergence = 'marginal', 'chi^2'

start = time.time()
__ = inference.variational_importance_sampling(inf_model, vari_model, inf_optimizer, vari_optimizer, train_dataloader, n_epochs, n_monte_carlo, update, divergence, print_freq)
end = time.time()


torch.save(inf_model.state_dict(), f'model/rgc_{method}_{n_hid_neurons}_{seed}_inf.pt')
torch.save(vari_model.state_dict(), f'model/rgc_{method}_{n_hid_neurons}_{seed}_vari.pt')

## evaluate
df = inference.evaluate_rgc(inf_model, vari_model, vis_spikes_list_test, convolved_vis_spikes_list_test, n_monte_carlo).mean().to_frame().T
df['time'] = end - start
df.to_csv(f'csv/rgc_{method}_{n_hid_neurons}_{seed}.csv', index=False)
