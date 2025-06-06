import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self, input_size, output_size, neurons, hidden_layers, Boussinesq, domain_points, ic_points, adapt_every=500):
        super(PINN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.epoch = 1
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = neurons
        self.hidden_layers = hidden_layers
        self.domain_points = domain_points
        self.ic_points = ic_points
        self.adapt_every = adapt_every

        self.boussinesq = Boussinesq
        
        self.model_snapshots = []

        layers = [nn.Linear(input_size, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons, device=self.device, dtype=self.dtype), nn.Tanh()]
        layers.append(nn.Linear(neurons, output_size, device=self.device, dtype=self.dtype))

        self.nn = nn.Sequential(*layers)

        self.nn.to(device=self.device, dtype=self.dtype)

        for layer in self.nn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.0)

        self.N_D, self.N_0 = adaptive_sampler(
            self, Boussinesq, domain_points, ic_points,
            dtype=self.dtype, device=self.device
        )

        self.optimizer = optim.Adam(self.nn.parameters(), lr=1e-3)

    def forward(self, x, t):
        input = torch.cat([x, t], dim=1)
        output = self.nn(input)
        eta = output[:, [0]] # shape = [N, 1]
        u = output[:, [1]]   # shape = [N, 1] por causa do autograd, temos que manter os tamanhos 2D (x,t) -> (eta, u)
        return eta, u

    def loss(self, Boussinesq, return_components=False):
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

    def train(self, Boussinesq, adapt_every=500, epochs=1000):
        self.adapt_every = adapt_every

        _, loss_ic, loss_Boussinesq = self.loss(Boussinesq, return_components=True)

        self.model_snapshots[-1].update(
            {
                "epoch": self.epoch,
                "N_D": self.N_D,
                "loss_ic": loss_ic.item(),
                "loss_Boussinesq": loss_Boussinesq.item()
            }
        )

        while self.epoch <= epochs:
            if self.epoch % self.adapt_every == 0:
                print(f"[{self.epoch}] Adaptive sampling of N_D...")

                self.N_D, self.N_0 = adaptive_sampler(
                    self, Boussinesq,
                    domain_points=self.domain_points,
                    ic_points=self.ic_points,
                    dtype=self.dtype, device=self.device
                )

                _, loss_ic, loss_Boussinesq = self.loss(Boussinesq, return_components=True)

                self.model_snapshots[-1].update(
                    {
                        "epoch": self.epoch,
                        "N_D": self.N_D,
                        "loss_ic": loss_ic.item(),
                        "loss_Boussinesq": loss_Boussinesq.item()
                    }
                )


            self.optimizer.zero_grad() # zera gradientes
            loss = self.loss(Boussinesq)
            loss.backward() # calcula d(loss)/dtheta
            self.optimizer.step() # da o passo do otimizador

            if self.epoch % 100 == 0:
                print(f"[{self.epoch}] Loss: {loss.item():.4e}")

            self.epoch += 1 

def adaptive_sampler(model, Boussinesq, domain_points, ic_points,
                     grid_factor=5, dtype=torch.float32, device='cpu',
                     k=1, c=0.25):

    # condição inicial com sampling uniforme
    x_0 = torch.empty(ic_points, 1, dtype=dtype, device=device).uniform_(Boussinesq.domain['x_min'], Boussinesq.domain['x_max'])
    t_0 = torch.full((ic_points, 1), Boussinesq.domain['t_min'], dtype=dtype, device=device)
    N_0 = torch.cat((x_0, t_0), dim=1)

    large_N = grid_factor * domain_points
    x_large = torch.empty(large_N, 1, dtype=dtype, device=device).uniform_(Boussinesq.domain['x_min'], Boussinesq.domain['x_max']).requires_grad_()
    t_large = torch.empty(large_N, 1, dtype=dtype, device=device).uniform_(Boussinesq.domain['t_min'], Boussinesq.domain['t_max']).requires_grad_()

    res_eq_1, res_eq_2 = Boussinesq.residual(model, x_large, t_large)

    # Aplicar a operação squeeze() nos resíduos
    res_eq_1 = res_eq_1.squeeze()  # Resíduo da primeira equação
    res_eq_2 = res_eq_2.squeeze()  # Resíduo da segunda equação

    # Calcular o erro absoluto elevado a k
    eps_k = (res_eq_1.abs() + res_eq_2.abs()).pow(k)  # |ε(x)|^k, somando os resíduos das duas equações

    if model.epoch % model.adapt_every == 0 or model.epoch == 1:
      model.model_snapshots.append({
          "loss_grid": eps_k.view(grid_factor * domain_points, 1).detach().cpu()
      })

    # RAD distribution
    E_eps_k = eps_k.mean()
    prob_dist = (eps_k / E_eps_k + c).detach()
    prob_dist /= prob_dist.sum()  # normaliza para virar uma probability density
    # OBS: p/ valores mto baixos da loss, parece que o valor de k muito alto acarreta em instabilidade numerica (gerando picos de probabilidades)
    # k = 1 gera menos instabilidade onde é quase zero o erro, assim, vira uma distr uniforme em c.

    idx = torch.multinomial(prob_dist, domain_points, replacement=True)
    x_d = x_large[idx].detach()
    t_d = t_large[idx].detach()
    N_D = torch.cat((x_d, t_d), dim=1)

    return N_D, N_0