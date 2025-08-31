import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self, input_size, output_size, neurons, hidden_layers, Boussinesq, 
                 domain_points, ic_points, optimizer_name='Adam', lr=1e-3, 
                 strategy='rar', N_add=100, adapt_every=500):
        super(PINN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        # --- Parâmetros de controle ---
        self.epoch = 1
        self.strategy = strategy.lower()
        self.N_add = N_add
        self.adapt_every = adapt_every
        self.domain_points_initial = domain_points
        self.ic_points = ic_points
        self.boussinesq = Boussinesq
        self.model_snapshots = []

        if self.strategy not in ['rar', 'rad']:
            raise ValueError("Estratégia deve ser 'rar' (refinamento) ou 'rad' (resampling).")

        # --- Arquitetura da Rede ---
        layers = [nn.Linear(input_size, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, output_size))
        self.nn = nn.Sequential(*layers).to(device=self.device, dtype=self.dtype)

        for layer in self.nn:
           if isinstance(layer, nn.Linear):
              nn.init.xavier_uniform_(layer.weight)
              layer.bias.data.fill_(0.0)

        # --- Amostragem Inicial (SEMPRE UNIFORME) ---
        self._initial_uniform_sampling()

        # --- Seleção do Otimizador ---
        self.optimizer_name = optimizer_name
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_name.lower() == 'l-bfgs':
            self.optimizer = optim.LBFGS(self.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"Otimizador '{optimizer_name}' não é suportado.")

    def _initial_uniform_sampling(self):
        """Cria o conjunto inicial de pontos de forma uniforme."""
        # Pontos da Condição Inicial
        x_0 = torch.empty(self.ic_points, 1, dtype=self.dtype, device=self.device).uniform_(
            self.boussinesq.domain['x_min'], self.boussinesq.domain['x_max']
        )
        t_0 = torch.full((self.ic_points, 1), self.boussinesq.domain['t_min'], dtype=self.dtype, device=self.device)
        self.N_0 = torch.cat((x_0, t_0), dim=1)

        # Pontos do Domínio (resíduo da EDP)
        x_d = torch.empty(self.domain_points_initial, 1, dtype=self.dtype, device=self.device).uniform_(
            self.boussinesq.domain['x_min'], self.boussinesq.domain['x_max']
        )
        t_d = torch.empty(self.domain_points_initial, 1, dtype=self.dtype, device=self.device).uniform_(
            self.boussinesq.domain['t_min'], self.boussinesq.domain['t_max']
        )
        self.N_D = torch.cat((x_d, t_d), dim=1)
        self.N_D_initial = self.N_D.clone() # para análise posterior
        print(f"Amostragem inicial: {self.N_D.shape[0]} pontos de domínio e {self.N_0.shape[0]} pontos de CI.")

    def _update_collocation_points(self, k=2, c=0.0):
        """Atualiza os pontos de colocação usando a estratégia RAR ou RAD."""
        
        # Define quantos pontos novos encontrar
        if self.strategy == 'rar':
            num_new_points = self.N_add
            # O conjunto de candidatos pode ser maior para ter mais variedade
            num_candidates = 10 * num_new_points 
        else: # strategy == 'rad'
            num_new_points = self.domain_points_initial
            num_candidates = 10 * num_new_points

        # Gera candidatos uniformemente
        x_cand = torch.empty(num_candidates, 1, dtype=self.dtype, device=self.device).uniform_(
            self.boussinesq.domain['x_min'], self.boussinesq.domain['x_max']
        ).requires_grad_()
        t_cand = torch.empty(num_candidates, 1, dtype=self.dtype, device=self.device).uniform_(
            self.boussinesq.domain['t_min'], self.boussinesq.domain['t_max']
        ).requires_grad_()
        
        # Calcula o resíduo e a distribuição de probabilidade
        res_eq_1, res_eq_2 = self.boussinesq.residual(self, x_cand, t_cand)
        res_norm = res_eq_1.abs().squeeze() + res_eq_2.abs().squeeze()
        prob_dist = (res_norm.pow(k) / res_norm.pow(k).mean() + c).detach()
        prob_dist /= prob_dist.sum()
        
        # Amostra os novos pontos de alta importância
        idx = torch.multinomial(prob_dist, num_new_points, replacement=True)
        new_points = torch.cat((x_cand[idx].detach(), t_cand[idx].detach()), dim=1)

        # Aplica a estratégia escolhida
        if self.strategy == 'rar':
            self.N_D = torch.cat([self.N_D, new_points], dim=0)
            print(f"Estratégia RAR: Adicionados {num_new_points} pontos. Total agora: {self.N_D.shape[0]}")
        else: # strategy == 'rad'
            self.N_D = new_points
            print(f"Estratégia RAD: Todos os {self.N_D.shape[0]} pontos foram reamostrados.")
            
    def forward(self, x, t):
        input_tensor = torch.cat([x, t], dim=1)
        output = self.nn(input_tensor)
        return output[:, [0]], output[:, [1]] # eta, u

    def loss(self, Boussinesq, return_components=False):
        # ... (seu código de loss, sem alterações) ...
        x_d, t_d = self.N_D[:, [0]], self.N_D[:, [1]]
        x_0, t_0 = self.N_0[:, [0]], self.N_0[:, [1]]

        res_eq_1, res_eq_2 = Boussinesq.residual(self, x_d, t_d)
        loss_Boussinesq = torch.mean(res_eq_1**2 + res_eq_2**2)

        u_pred_0, eta_pred_0 = self.forward(x_0, t_0)
        u_true_0, eta_true_0 = Boussinesq.ic(x_0)
        loss_ic = torch.mean((u_pred_0 - u_true_0)**2 + (eta_pred_0 - eta_true_0)**2)

        loss = loss_Boussinesq + loss_ic

        if return_components:
            return loss, loss_Boussinesq, loss_ic

        return loss

    def run_train_loop(self, Boussinesq, epochs=1000):
        print(f"Iniciando treino com Otimizador: {self.optimizer_name.upper()}, Estratégia: {self.strategy.upper()}")

        if self.optimizer_name.lower() == 'l-bfgs':
            def closure():
                self.optimizer.zero_grad()
                loss = self.loss(Boussinesq)
                loss.backward()
                return loss
            
            for epoch in range(1, epochs + 1):
                self.epoch = epoch
                # A amostragem adaptativa pode ser usada com L-BFGS, mas é menos comum
                if self.epoch % self.adapt_every == 0:
                    self._update_collocation_points()
                
                self.optimizer.step(closure)
                loss = self.loss(Boussinesq)
                print(f"[{self.epoch}] Loss: {loss.item():.4e}")
        else: # Adam
            for epoch in range(1, epochs + 1):
                self.epoch = epoch
                if self.epoch % self.adapt_every == 0:
                    self._update_collocation_points()

                self.optimizer.zero_grad()
                loss = self.loss(Boussinesq)
                loss.backward()
                self.optimizer.step()

                if self.epoch % 100 == 0:
                    print(f"[{self.epoch}] Loss: {loss.item():.4e}")