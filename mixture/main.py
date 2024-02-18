import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

import argparse

import model, inference

## arguments
parser = argparse.ArgumentParser()
parser.add_argument('idx', type=int)
args = parser.parse_args()

method_list = ['VI', 'CHIVI', 'VBIS', 'VIS']
seed_list = np.arange(10)

arg_index = np.unravel_index(args.idx, (len(method_list), len(seed_list)))
method, seed = method_list[arg_index[0]], seed_list[arg_index[1]]


## hyper-parameters
torch.manual_seed(0)
pi = torch.tensor(0.7)
n_samples = 1000
gen_model = model.GMM4Bernoulli(pi, torch.tensor([-8, -2, 2, 8.])) # generative model
z_list_train, x_list_train = gen_model.sample(n_samples) # generate hidden and observed training data
z_list_test, x_list_test = gen_model.sample(n_samples) # generate hidden and observed test data
n_epochs = 200
print_freq = 10
n_monte_carlo = 5000
n_batches = 10
batch_size = int(n_samples / n_batches)
grad = 'score'


torch.manual_seed(seed)
inf_model = model.GMM4Bernoulli(torch.tensor(0.5), torch.tensor([-9, -1, 1, 9.]))
vari_model = model.Post2Gaussian(torch.tensor([-9, 9.]), torch.tensor([1, 1.]))

inf_optimizer = torch.optim.Adam(inf_model.parameters(), lr=0.002)
vari_optimizer = torch.optim.Adam(vari_model.parameters(), lr=0.002)


if method == 'VI':
    update, divergence = 'ELBO', 'RKL'
elif method == 'CHIVI':
    update, divergence = 'ELBO', 'sandwich'
elif method == 'VBIS':
    update, divergence = 'marginal', 'RKL'
elif method == 'VIS':
    update, divergence = 'marginal', 'chi^2'

start = time.time()
__, record_mu, record_pi = inference.variational_importance_sampling(inf_model, vari_model, inf_optimizer, vari_optimizer, x_list_train, n_epochs, n_monte_carlo, batch_size, grad, update, divergence, print_freq)
end = time.time()


torch.save(inf_model.state_dict(), f'model/{method}_{seed}_inf.pt')
torch.save(vari_model.state_dict(), f'model/{method}_{seed}_vari.pt')
np.save(f'np/{method}_{seed}_mu.npy', record_mu)
np.save(f'np/{method}_{seed}_pi.npy', record_pi)


## evaluate
df = inference.evaluate(inf_model, vari_model, x_list_test, z_list_test, n_monte_carlo, seed).mean().to_frame().T
df['time'] = end - start
df.to_csv(f'csv/{method}_{seed}.csv', index=False)
