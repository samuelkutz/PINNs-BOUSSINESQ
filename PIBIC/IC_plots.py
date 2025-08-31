import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath('./PINN'))
sys.path.append(os.path.abspath('./BOUSSINESQ'))

from PINN.PINN import PINN
from BOUSSINESQ.BOUSSINESQ import Boussinesq
def plot_rar_analysis():
    os.makedirs('./IC_plots', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    alpha = beta = 0.1
    pinn_params = { "input_size": 2, "output_size": 2, "neurons": 50, "hidden_layers": 4, "domain_points": 5000, "ic_points": 200, "strategy": "rar" }

    # --- Carregar Modelos e Pontos ---
    b_pinn = Boussinesq(-30., 30., 0., 15., alpha, beta)
    
    # Modelo com RAR
    model_rar = PINN(Boussinesq=b_pinn, **pinn_params).to(device)
    model_rar.load_state_dict(torch.load(f'./PINN/pinn_solution_{alpha}_{beta}.pth', map_location=device))
    model_rar.eval()
    points_initial = model_rar.N_D_initial.cpu()
    points_final_rar = torch.load(f'./PINN/collocation_points_{alpha}_{beta}.pt', map_location='cpu')

    # Modelo sem RAR
    model_no_rar = PINN(Boussinesq=b_pinn, **pinn_params).to(device)
    model_no_rar.load_state_dict(torch.load(f'./PINN/pinn_solution_{alpha}_{beta}_no_rar.pth', map_location=device))
    model_no_rar.eval()

    # --- Carregar Referência ---
    x_ref, t_ref, etas_ref, _ = torch.load(f'./BOUSSINESQ/pseudo_spectral_solution_{alpha}_{beta}.pt', map_location=device)
    etas_ref = torch.stack(etas_ref) if isinstance(etas_ref, list) else etas_ref
    t_final_val = t_ref[-1].item()

    # --- Gerar Predições ---
    x_tensor = x_ref.clone().detach().reshape(-1, 1)
    t_final_tensor = torch.full_like(x_tensor, t_final_val)
    with torch.no_grad():
        eta_pinn_rar, _ = model_rar(x_tensor, t_final_tensor)
        eta_pinn_no_rar, _ = model_no_rar(x_tensor, t_final_tensor)

    # --- Criar Plots (Layout 3x2) ---
    fig = plt.figure(figsize=(14, 15), constrained_layout=True)
    fig.suptitle(f'Análise do Efeito da Amostragem Adaptativa (RAR) para $\\alpha=\\beta={alpha}$', fontsize=18, weight='bold')

    gs = fig.add_gridspec(3, 2)

    # Linha 1, Coluna 1: Distribuição Inicial de Pontos
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(points_initial[:, 0], points_initial[:, 1], s=1, alpha=0.5)
    ax1.set_title('Distribuição Inicial de Pontos (Uniforme)')
    ax1.set_xlabel('x'); ax1.set_ylabel('t')

    # Linha 1, Coluna 2: Distribuição Final com RAR
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(points_final_rar[:, 0], points_final_rar[:, 1], s=1, alpha=0.5, c='r')
    ax2.set_title('Distribuição Final de Pontos (com RAR)')
    ax2.set_xlabel('x'); ax2.set_ylabel('t')

    # Linha 2: Comparação no Domínio Físico
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(x_ref.cpu(), etas_ref[-1].cpu(), 'k-', label='Referência', lw=2)
    ax3.plot(x_ref.cpu(), eta_pinn_no_rar.cpu(), 'b--', label='PINN (Uniforme)', lw=2)
    ax3.plot(x_ref.cpu(), eta_pinn_rar.cpu(), 'r-.', label='PINN (com RAR)', lw=2)
    ax3.set_title(f'Solução no Tempo Final (t={t_final_val:.1f})')
    ax3.set_xlabel('x'); ax3.set_ylabel('$\\eta(x,t)$')
    ax3.legend(); ax3.grid(True, alpha=0.5)

    # Linha 3: Espectro de Fourier (Comparação)
    ax4 = fig.add_subplot(gs[2, :])
    N, dx = len(x_ref), (x_ref[1] - x_ref[0]).item()
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    k_half = k[:N//2]
    
    fft_ref = np.abs(np.fft.fft(etas_ref[-1].cpu().numpy()))[:N//2]
    fft_rar = np.abs(np.fft.fft(eta_pinn_rar.cpu().numpy().flatten()))[:N//2]
    fft_no_rar = np.abs(np.fft.fft(eta_pinn_no_rar.cpu().numpy().flatten()))[:N//2]

    ax4.semilogy(k_half, fft_ref, 'k-', label='Referência', lw=2)
    ax4.semilogy(k_half, fft_no_rar, 'b--', label='PINN (Uniforme)', lw=2)
    ax4.semilogy(k_half, fft_rar, 'r-.', label='PINN (com RAR)', lw=2)
    ax4.set_title('Espectro de Fourier da Solução Final')
    ax4.set_xlabel('Número de Onda (k)'); ax4.set_ylabel('Amplitude')
    ax4.legend(); ax4.grid(True, which="both", alpha=0.5); ax4.set_xlim(0, 10)

    plt.savefig('./IC_plots/rar_ablation_study_panel.png', dpi=300)
    plt.close(fig)

def plot_main_comparison():
    os.makedirs('./IC_plots', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cases = [1e-4, 1e-3, 1e-2, 1e-1]
    pinn_params = { "input_size": 2, "output_size": 2, "neurons": 50, "hidden_layers": 4, "domain_points": 1, "ic_points": 1, "strategy": "rar" }

    # --- Cria uma única figura para todos os casos (painel) ---
    fig, axs = plt.subplots(len(cases), 3, figsize=(18, 8), constrained_layout=True, sharex='col')
    fig.suptitle('Comparação PINN vs. Pseudoespectral para o Sistema de Boussinesq', fontsize=18, weight='bold')

    for i, val in enumerate(cases):
        alpha = beta = val
        print(f"Processando painel para alpha = beta = {val}...")

        # --- Carregar Dados ---
        ref_path = f'./BOUSSINESQ/pseudo_spectral_solution_{alpha}_{beta}.pt'
        model_path = f'./PINN/pinn_solution_{alpha}_{beta}.pth'
        
        try:
            x_ref, t_ref, etas_ref, _ = torch.load(ref_path, map_location=device)
            etas_ref = torch.stack(etas_ref) if isinstance(etas_ref, list) else etas_ref
            
            b_pinn = Boussinesq(-30., 30., 0., 15., alpha, beta)
            model = PINN(Boussinesq=b_pinn, **pinn_params).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        except FileNotFoundError as e:
            print(f"AVISO: Arquivo não encontrado para o caso alpha=beta={val}. Pulando. ({e})")
            for j in range(3):
                axs[i, j].text(0.5, 0.5, f'Dados para α=β={val} não encontrados', ha='center', va='center')
                axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            continue

        # --- Gerar Predições ---
        x_tensor = x_ref.clone().detach().reshape(-1, 1)
        t0_tensor = torch.zeros_like(x_tensor)
        t_final_val = t_ref[-1].item()
        t_final_tensor = torch.full_like(x_tensor, t_final_val)

        with torch.no_grad():
            eta_pinn_t0, _ = model(x_tensor, t0_tensor)
            eta_pinn_t_final, _ = model(x_tensor, t_final_tensor)
        
        # --- Coluna 1: Condição Inicial ---
        ax = axs[i, 0]
        ax.plot(x_ref.cpu(), etas_ref[0].cpu(), 'k-', label='Referência (Exata)', lw=2)
        ax.plot(x_ref.cpu(), eta_pinn_t0.cpu(), 'r--', label='PINN', lw=2)
        ax.set_ylabel(f'$\\alpha=\\beta={val:g}$\n\n$\\eta(x,0)$', fontsize=12, rotation=0, labelpad=40, va='center')
        ax.grid(True, alpha=0.5)
        if i == 0:
            ax.set_title('Condição Inicial (t=0.0)', fontsize=14)
            ax.legend()

        # --- Coluna 2: Tempo Final ---
        ax = axs[i, 1]
        ax.plot(x_ref.cpu(), etas_ref[-1].cpu(), 'k-', label='Referência (Pseudo.)', lw=2)
        ax.plot(x_ref.cpu(), eta_pinn_t_final.cpu(), 'r--', label='PINN', lw=2)
        ax.set_ylabel('$\\eta(x, t_{final})$', fontsize=12)
        ax.grid(True, alpha=0.5)
        if i == 0:
            ax.set_title(f'Tempo Final (t={t_final_val:.1f})', fontsize=14)
            ax.legend()

        # --- Coluna 3: Espectro de Fourier ---
        ax = axs[i, 2]
        N, dx = len(x_ref), (x_ref[1] - x_ref[0]).item()
        k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        k_half = k[:N//2]
        
        fft_ref = np.abs(np.fft.fft(etas_ref[-1].cpu().numpy()))[:N//2]
        fft_pinn = np.abs(np.fft.fft(eta_pinn_t_final.cpu().numpy().flatten()))[:N//2]

        ax.semilogy(k_half, fft_ref, 'k-', label='Referência (Pseudo.)', lw=2)
        ax.semilogy(k_half, fft_pinn, 'r--', label='PINN', lw=2)
        ax.set_xlim(0, 10)
        ax.set_ylabel('Amplitude Normalizada', fontsize=12)
        ax.grid(True, which="both", alpha=0.5)
        if i == 0:
            ax.set_title(f'Espectro de Fourier (t={t_final_val:.1f})', fontsize=14)
            ax.legend()

    # --- Ajustes Finais e Salvamento ---
    for i in range(len(cases)):
        axs[i, 1].yaxis.set_label_position("right")
        axs[i, 2].yaxis.set_label_position("right")

    axs[-1, 0].set_xlabel('x', fontsize=12)
    axs[-1, 1].set_xlabel('x', fontsize=12)
    axs[-1, 2].set_xlabel('Número de Onda (k)', fontsize=12)

    plt.savefig('./IC_plots/boussinesq_comparison_panel.png', dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    plot_main_comparison()
    plot_rar_analysis()