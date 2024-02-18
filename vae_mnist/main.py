import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

import argparse

import model, inference

## arguments
parser = argparse.ArgumentParser()
parser.add_argument('idx', type=int)
args = parser.parse_args()

method_list = ['VI', 'CHIVI', 'VBIS', 'VIS', 'IWAE']
seed_list = np.arange(5)
arg_index = np.unravel_index(args.idx, (len(method_list), len(seed_list)))
method, seed = method_list[arg_index[0]], seed_list[arg_index[1]]


training_data = datasets.MNIST(
    root="../vae_mnist/data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="../vae_mnist/data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_data.data.to(torch.device('cuda:0'))
training_data.targets.to(torch.device('cuda:0'))
test_data.data.to(torch.device('cuda:0'))
test_data.targets.to(torch.device('cuda:0'))

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

torch.manual_seed(seed)

n_epochs = 20
print_freq = 1
n_monte_carlo = 500


inf_model = model.Decoder().to(training_data.data.device)
vari_model = model.Encoder().to(training_data.data.device)

inf_optimizer = torch.optim.Adam(inf_model.parameters(), lr=5e-3)
vari_optimizer = torch.optim.Adam(vari_model.parameters(), lr=5e-3)

if method == 'VI':
    grad, update, divergence = 'pathwise', 'ELBO', 'RKL'
elif method == 'CHIVI':
    grad, update, divergence = 'score', 'ELBO', 'sandwich'
elif method == 'VBIS':
    grad, update, divergence = 'pathwise', 'marginal', 'RKL'
elif method == 'VIS':
    grad, update, divergence = 'score', 'marginal', 'chi^2'
elif method == 'IWAE':
    grad, update, divergence = 'pathwise', 'marginal', 'importance'

start = time.time()
__ = inference.variational_importance_sampling(inf_model, vari_model, inf_optimizer, vari_optimizer, train_dataloader, test_data, method, seed, n_epochs, n_monte_carlo, grad, update, divergence, print_freq)
end = time.time()


torch.save(inf_model.state_dict(), f'model/{method}_{seed}_inf.pt')
torch.save(vari_model.state_dict(), f'model/{method}_{seed}_vari.pt')

## evaluate
df = inference.evaluate(inf_model, vari_model, test_data, 1000).mean().to_frame().T
df['time'] = end - start
df.to_csv(f'csv/{method}_{seed}.csv', index=False)
