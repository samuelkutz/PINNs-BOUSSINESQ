# OBS: este código é bem único e não muito adaptável para outras EDPs

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

neuron = 6
layer = 3

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

boussinesq = Boussinesq(x_min=-30.0, x_max=30.0, t_min=0.0, t_max=15.0, a=1e-1, b=1e-1)

model = PINN(
    input_size=2, output_size=2, neurons=50, hidden_layers=4,
    Boussinesq=boussinesq,
    domain_points=5000,
    ic_points=200,
    optimizer_name="adam",
    strategy="rar",      # <-- ESCOLHE A ESTRATÉGIA
    N_add=500,           # <-- Número de pontos a adicionar
    adapt_every=500      # <-- Frequência
)
model.train(boussinesq, epochs=15000)


torch.save(model, f'./PINN/pinn_solution_{boussinesq.a}_{boussinesq.b}.pth')