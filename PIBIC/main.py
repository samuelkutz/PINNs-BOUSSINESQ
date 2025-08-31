import torch
import os
from concurrent.futures import ProcessPoolExecutor
import sys

# Adicionando os caminhos para os módulos customizados
sys.path.append(os.path.abspath('./PINN'))
sys.path.append(os.path.abspath('./BOUSSINESQ'))

from PINN.PINN import PINN
from BOUSSINESQ.BOUSSINESQ import Boussinesq
from BOUSSINESQ.PseudoSpectral import PseudoSpectralBoussinesq

# --- CONFIGURAÇÃO DOS EXPERIMENTOS ---
# Cada tupla é (valor_alpha_beta, usar_rar, nome_sufixo)
experiments_config = [
    (0.0,    True, ""),
    (1e-3,   True, ""),
    (1e-1,   True, ""),
    (1e-1,   False, "_no_rar") # Caso extra para análise de ablação
]

# Hiperparâmetros da PINN
pinn_params_base = {
    "input_size": 2,
    "output_size": 2,
    "neurons": 50,
    "hidden_layers": 4,
    "domain_points": 5000,
    "ic_points": 200,
    "optimizer_name": "adam",
    "strategy": "rar",
    "N_add": 500,
}
epochs = 15000

def run_experiment(config):
    """
    Executa um experimento completo para uma dada configuração.
    """
    val, use_rar, suffix = config
    alpha = val
    beta = val
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    seed = 39
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pid = os.getpid()
    print(f"--- [Processo {pid}] Iniciando: alpha={alpha}, beta={beta}, RAR={use_rar} ---")

    # --- Gerar solução de referência (apenas se não existir) ---
    ref_filename = f'./BOUSSINESQ/pseudo_spectral_solution_{alpha}_{beta}.pt'
    if not os.path.exists(ref_filename):
        print(f"--- [Processo {pid}] Gerando solução de referência...")
        boussinesq_ref = Boussinesq(x_min=-30.0, x_max=30.0, t_min=0.0, t_max=15.0, a=alpha, b=beta)
        ps = PseudoSpectralBoussinesq(boussinesq_ref, Nx=256, Nt=5000, device=device)
        x, t, etas, us = ps.solve(save_every=25)
        torch.save((x, t, etas, us), ref_filename)
        print(f"--- [Processo {pid}] Solução de referência salva.")

    # --- Configurar e treinar a PINN ---
    pinn_params = pinn_params_base.copy()
    if use_rar:
        pinn_params["adapt_every"] = 500
    else:
        # Desativa o RAR definindo a adaptação para depois do fim do treino
        pinn_params["adapt_every"] = epochs + 1 

    print(f"--- [Processo {pid}] Configurando e treinando a PINN...")
    boussinesq_pinn = Boussinesq(x_min=-30.0, x_max=30.0, t_min=0.0, t_max=15.0, a=alpha, b=beta)
    model = PINN(Boussinesq=boussinesq_pinn, **pinn_params).to(device)
    model.run_train_loop(boussinesq_pinn, epochs=epochs)

    # --- Salvar o modelo e os pontos de colocação finais ---
    model_suffix = f"{alpha}_{beta}{suffix}"
    model_path = f'./PINN/pinn_solution_{model_suffix}.pth'
    points_path = f'./PINN/collocation_points_{model_suffix}.pt'
    
    torch.save(model.state_dict(), model_path)
    torch.save(model.N_D, points_path) # Salva o tensor de pontos final
    
    return f"--- [Processo {pid}] Concluído: {model_suffix}. Modelo e pontos salvos. ---"

if __name__ == '__main__':
    os.makedirs('./BOUSSINESQ', exist_ok=True)
    os.makedirs('./PINN', exist_ok=True)

    max_workers = os.cpu_count() 
    print(f"Iniciando bateria de testes com até {max_workers} processos paralelos.")

    # Usando map para rodar em paralelo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_experiment, experiments_config)
        for result in results:
            print(result)

    print("\nBateria de experimentos finalizada.")