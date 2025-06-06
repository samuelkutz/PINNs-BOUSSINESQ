# OBS: para rodar tem q usar python -m BOUSSINESQ.generate_solution com o -m pois nao estamos em root (main.py)

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

neuron = 8
layer = 2

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA')
else:
    device = torch.device('cpu')    
    print('CPU')

seed = 39
torch.manual_seed(seed) # CPU
torch.cuda.manual_seed_all(seed) # GPUs

print(f"Using {device} enviroment.")

import sys
import os
sys.path.append(os.path.abspath('./PINN'))
sys.path.append(os.path.abspath('./BOUSSINESQ'))

from PINN import PINN
from BOUSSINESQ import Boussinesq
from PseudoSpectral import PseudoSpectralBoussinesq

boussinesq = Boussinesq(x_min=-30.0, x_max=30.0, t_min=0.0, t_max=15.0, a=1e-1, b=1e-1)

ps = PseudoSpectralBoussinesq(boussinesq, Nx = 256, Nt = 5000)

x, t, etas, us = ps.solve(save_every=25)

data = x, t, etas, us

torch.save(data, f'./BOUSSINESQ/pseudo_spectral_solution_{boussinesq.a}_{boussinesq.b}.pt')
