import torch
import torch.nn as nn
import torch.optim as optim

# TODO: arrumar dtype e device (funciona sem)
class Boussinesq:
    def __init__(self, x_min, x_max, t_min, t_max, dtype=torch.float32, device='cpu', a=0, b=0, F=0, A=1):
        self.A = A

        self.a = a
        self.b = b
        self.F = F
        self.domain = {
            'x_min': torch.tensor(x_min, dtype=dtype, device=device),
            'x_max': torch.tensor(x_max, dtype=dtype, device=device),
            't_min': torch.tensor(t_min, dtype=dtype, device=device),
            't_max': torch.tensor(t_max, dtype=dtype, device=device),
        }

    def ic(self, x):
        k = torch.tensor(1, dtype=x.dtype, device=x.device)
        eta0 = self.A / torch.cosh(k * x) ** 2
        u0 = torch.zeros_like(x, dtype=x.dtype, device=x.device)  # vetor de zeros do mesmo tamanho que x
        return eta0, u0
    
    def residual(self, model, x, t):    
        x.requires_grad_(True)
        t.requires_grad_(True)

        eta, u = model(x, t)

        # autograd é o papo
        eta_t = torch.autograd.grad(eta, t, grad_outputs=torch.ones_like(eta), retain_graph=True, create_graph=True)[0]
        eta_x = torch.autograd.grad(eta, x, grad_outputs=torch.ones_like(eta), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_xxt = torch.autograd.grad(u_xx, t, grad_outputs=torch.ones_like(u_xx), retain_graph=True, create_graph=True)[0]

        res_eq_1 = eta_t + u_x + self.a*(eta_x*u + eta*u_x)
        res_eq_2 = u_t + eta_x + self.a*u*u_x - (self.b/3)*u_xxt

        return res_eq_1, res_eq_2