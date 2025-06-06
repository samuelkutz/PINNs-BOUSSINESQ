import torch

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter

from PINN.PINN import PINN
from BOUSSINESQ.BOUSSINESQ import Boussinesq

# Modelo PINN treinado
model_linear = torch.load('./PINN/pinn_solution_0_0.pth', weights_only=False)

# Carrega x, t da simulação pseudoespectral
x, t, _, _ = torch.load('./BOUSSINESQ/pseudo_spectral_solution_0.1_0.1.pt')

X = x.cpu().numpy()
T = np.array(t)

# Gera grade 2D
X_grid, T_grid = np.meshgrid(X, T, indexing='ij')  # X: (Nx, Nt)

# Instancia equação com a=b=0
boussinesq = Boussinesq(x_min=X[0], x_max=X[-1], t_min=T[0], t_max=T[-1], A=1.0, a=0.0, b=0.0)

# Condição inicial como função numpy pura
def eta0_np(x):
    k = 1.0
    A = boussinesq.A
    return A / np.cosh(k * x) ** 2

# Solução exata linear: soma de ondas para esquerda e direita
ETA_exact = 0.5 * eta0_np(X_grid - T_grid) + 0.5 * eta0_np(X_grid + T_grid)

# === PLOT COMPARATIVO COM A PINN ===
indices = [0, -1]
titles = ['Época Inicial', 'Época Final']
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

for i, idx in enumerate(indices):
    snapshot = model_linear.model_snapshots[idx]
    epoch = snapshot["epoch"]

    im = axes[i].imshow(
        ETA_exact.T,  # Transpor: t no eixo vertical
        extent=[X[0], X[-1], T[0], T[-1]],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )

    N_D = snapshot["N_D"]
    x_d = N_D[:, 0].cpu().numpy()
    t_d = N_D[:, 1].cpu().numpy()

    axes[i].scatter(
        x_d, t_d,
        c='red', s=10, alpha=0.9,
        edgecolors='black', linewidths=0.1,
        label='Colocação'
    )

    axes[i].set_title(f'{titles[i]} (Época {epoch})', fontsize=10)
    axes[i].set_xlabel('$x$')
    axes[i].label_outer()
    axes[i].legend(loc='upper right', fontsize=7)

axes[0].set_ylabel('$t$')
fig.suptitle(r'Comparação: Solução Analítica $\eta(x,t)$ vs  Pontos de Colocação da PINN' + '\n' + r'$(\alpha = 0, \beta = 0)$', fontsize=12)
plt.tight_layout()
plt.savefig(f'comparacao_eta_colocacao_linear.png', dpi=300)

################# PLOT BOUSS NAO LINEAR $$$$$$$$$$$$$$$ 

model_nonlinear = torch.load('./PINN/pinn_solution_0.1_0.1.pth', weights_only=False)

x, t, etas, us = torch.load('./BOUSSINESQ/pseudo_spectral_solution_0.1_0.1.pt')

X = x.cpu().numpy()
T = np.array(t)
ETA = torch.stack(etas).T.cpu().numpy()  # shape (Nx, Nt_saved)

# Seleciona duas épocas do PINN: início e fim
indices = [0, -1]
titles = ['Época Inicial', 'Época Final']
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

for i, idx in enumerate(indices):
    snapshot = model_nonlinear.model_snapshots[idx]
    epoch = snapshot["epoch"]

    # Heatmap da solução espectral com x no eixo horizontal e t no vertical
    im = axes[i].imshow(
        ETA.T,  # <-- transpor a matriz para alinhar eixos corretamente
        extent=[X[0], X[-1], T[0], T[-1]],  # x no eixo horizontal, t no vertical
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )

    # Pontos de colocação da PINN
    N_D = snapshot["N_D"]
    x_d = N_D[:, 0].cpu().numpy()
    t_d = N_D[:, 1].cpu().numpy()
    axes[i].scatter(
        x_d, t_d,
        c='red', s=10, alpha=0.9,
        edgecolors='black', linewidths=0.1,
        label='Colocação'
    )

    axes[i].set_title(f'{titles[i]} (Época {epoch})', fontsize=10)
    axes[i].set_xlabel('$x$')
    axes[i].label_outer()
    axes[i].legend(loc='upper right', fontsize=7)

axes[0].set_ylabel('$t$')
fig.suptitle('Comparação: Solução Espectral de $\\eta(x,t)$ e Pontos de Colocação da PINN', fontsize=12)
plt.tight_layout()
plt.savefig(f'comparacao_eta_colocacao_nonlinear.png', dpi=300)


######### PLOT ANIMACAO BOUSS LINEAR #########3

# Carrega grade e tempos
x, t_plot, _, _ = torch.load('./BOUSSINESQ/pseudo_spectral_solution_0.1_0.1.pt')
x = x.numpy()
t_plot = np.array(t_plot)

# Prepara entrada para a PINN
x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(model_linear.device)

def predict_eta(t_val):
    t_tensor = torch.full_like(x_tensor, t_val)
    eta_pred, _ = model_linear(x_tensor, t_tensor)
    return eta_pred.detach().cpu().numpy().flatten()

# Define função da condição inicial (igual à usada na classe Boussinesq)
def eta0_np(x):
    k = 1.0
    A = 1.0
    return A / np.cosh(k * x) ** 2

# Cria função da solução exata linearizada
def eta_linear_exact(x, t):
    return 0.5 * eta0_np(x - t) + 0.5 * eta0_np(x + t)

# ===== Criação do GIF =====
fig = plt.figure(figsize=(6, 4.5))
plt.title('PINN vs solução analítica — Boussinesq Linearizada')
plt.xlabel('x')
plt.ylabel(r'$\eta(x,t)$')
plt.grid(True)

line_pinn, = plt.plot([], [], 'b-', label='PINN')
line_exact, = plt.plot([], [], 'r--', label='Analítica')
plt.legend(loc='upper left')
plt.xlim(x.min(), x.max())
plt.ylim(-0.1, 1.0)

time_text = plt.figtext(0.15, 0.65, '', fontsize=10, backgroundcolor='lightgrey')

def update(frame):
    t_val = t_plot[frame]

    eta_pred = predict_eta(t_val)
    eta_ref = eta_linear_exact(x, t_val)

    line_pinn.set_data(x, eta_pred)
    line_exact.set_data(x, eta_ref)

    time_text.set_text(f't = {t_val:.2f}')
    return line_pinn, line_exact, time_text

anim = FuncAnimation(fig, update, frames=len(t_plot), interval=60)

writer = PillowWriter(fps=24, metadata=dict(artist='Samuel Kutz'), bitrate=1800)
anim.save(f'linear_pinn_vs_sol_analitica.gif', writer=writer)


######## PLOT ANIMACAO BOUSS NAO LINEAR ######

x, t_plot, eta_spectral, _ = torch.load('./BOUSSINESQ/pseudo_spectral_solution_0.1_0.1.pt')  # Ajuste o path se necessário
x = x.numpy()
t_plot = np.array(t_plot)
eta_spectral = torch.stack(eta_spectral).numpy() # shape: [Nt_plot, Nx]

# ===== Prepara entrada para a PINN =====
x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
x_tensor = x_tensor.to(model_nonlinear.device)

def predict_eta(t_val):
    t_tensor = torch.full_like(x_tensor, t_val)
    eta_pred, _ = model_nonlinear(x_tensor, t_tensor)
    return eta_pred.detach().cpu().numpy().flatten()

# ===== Criação do GIF =====
fig = plt.figure(figsize=(6, 4.5))
plt.title('PINN vs método pseudoespectral — Boussinesq')
plt.xlabel('x')
plt.ylabel(r'$\eta(x,t)$')
plt.grid(True)

line_pinn, = plt.plot([], [], 'b-', label='PINN')
line_spec, = plt.plot([], [], 'r--', label='Pseudoespectral')
plt.legend(loc='upper left')
plt.xlim(x.min(), x.max())
plt.ylim(-0.1, 1.0)

time_text = plt.figtext(0.15, 0.65, '', fontsize=10, backgroundcolor='lightgrey')

def update(frame):
    t_val = t_plot[frame]

    eta_pred = predict_eta(t_val)
    eta_ref = eta_spectral[frame]

    line_pinn.set_data(x, eta_pred)
    line_spec.set_data(x, eta_ref)

    time_text.set_text(f't = {t_val:.2f}')
    return line_pinn, line_spec, time_text

anim = FuncAnimation(fig, update, frames=(len(t_plot)), interval=60)

writer = PillowWriter(fps=24, metadata=dict(artist='Samuel Kutz'), bitrate=1800)
anim.save(f'nonlinear_pinn_vs_pseudoespectral.gif', writer=writer)


########## PLOT FREQUENCIAS #######


# Último tempo do vetor t_plot
t_idx = -1
t_val = t_plot[t_idx]

# Solução pseudoespectral nesse tempo (torch tensor)
eta_spec = torch.tensor(eta_spectral[t_idx], dtype=torch.float32, device=model_nonlinear.device)  # [Nx]
# Solução da PINN nesse tempo (torch tensor)
eta_pinn = torch.tensor(predict_eta(t_val), dtype=torch.float32, device=model_nonlinear.device)   # [Nx]

N = eta_spec.shape[0]
dx = x[1] - x[0]

# FFT com PyTorch
fft_spec = torch.fft.fft(eta_spec)
fft_pinn = torch.fft.fft(eta_pinn)

# Frequências (número de onda)
freq = torch.fft.fftfreq(N, d=dx) * 2 * torch.pi

# Organiza para plotar: fftshift
def fftshift(x):
    n = x.shape[0]
    p2 = (n + 1) // 2
    return torch.cat((x[p2:], x[:p2]))

freq_shifted = fftshift(freq)
amp_spec = fftshift(torch.abs(fft_spec))
amp_pinn = fftshift(torch.abs(fft_pinn))

# Normaliza amplitudes
amp_spec /= amp_spec.max()
amp_pinn /= amp_pinn.max()

# Passa para numpy para plot
freq_np = freq_shifted.cpu().numpy()
amp_spec_np = amp_spec.cpu().numpy()
amp_pinn_np = amp_pinn.cpu().numpy()

plt.figure(figsize=(8, 4))
plt.plot(freq_np, amp_spec_np, 'r--', label='Pseudoespectral')
plt.plot(freq_np, amp_pinn_np, 'b-', label='PINN')
plt.yscale('log')
plt.xlim(0, 13)
plt.xlabel('Número de onda $k$')
plt.ylabel('Amplitude (log)')
plt.title(f'Espectro de $\\eta(x, t={t_val:.2f})$ (último tempo)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'espectro_fourier_solucoes_nonlinear.png', dpi=300)