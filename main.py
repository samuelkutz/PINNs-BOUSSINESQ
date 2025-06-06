# OBS: este código é bem único e não muito adaptável para outras EDPs

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

neuron = 12
layer = 4

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

from PINN.PINN import PINN
from BOUSSINESQ.BOUSSINESQ import Boussinesq

boussinesq = Boussinesq(x_min=-30.0, x_max=30.0, t_min=0.0, t_max=15.0, a=0, b=0)

# monta a PINN  
model = PINN(
    input_size=2, output_size=2, neurons=neuron, hidden_layers=layer,
    Boussinesq=boussinesq, domain_points=5000, ic_points=200
)

model.train(boussinesq, epochs=9999, adapt_every=1000)  # use menos épocas para teste rápido

torch.save(model, f'./PINN/pinn_solution_{boussinesq.a}_{boussinesq.b}.pth')