import numpy as np
import pandas as pd

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
trial_list = np.arange(10)
seed_list = np.arange(10)

arg_index = np.unravel_index(args.idx, (len(method_list), len(trial_list), len(seed_list)))
method, trial, seed = method_list[arg_index[0]], trial_list[arg_index[1]], seed_list[arg_index[2]]


decay = 5
dt = 0.05
window_size = 5
n_neurons = 5
n_vis_neurons = 3
basis = utils.exp_basis(decay, window_size, dt*window_size)
T = 5

n_epochs = 20
print_freq = 1
n_monte_carlo = 2000


df = pd.read_pickle('data.pkl')

spikes_list_train = df.at[trial, 'spikes_list_train']
convolved_spikes_list_train = df.at[trial, 'convolved_spikes_list_train']

train_dataset = TensorDataset(spikes_list_train[:, :, :n_vis_neurons], convolved_spikes_list_train[:, :, :n_vis_neurons])
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False)


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


torch.save(inf_model.state_dict(), f'model/{method}_{trial}_{seed}_inf.pt')
torch.save(vari_model.state_dict(), f'model/{method}_{trial}_{seed}_vari.pt')

## evaluate
df = inference.evaluate(inf_model, vari_model, df.at[trial, 'spikes_list_test'], df.at[trial, 'convolved_spikes_list_test'], n_monte_carlo).mean().to_frame().T
df['time'] = end - start
df.to_csv(f'csv/{method}_{trial}_{seed}.csv', index=False)
