import torch

class Boussinesq:
    """
    Define o problema do sistema de Boussinesq, incluindo domínio,
    parâmetros, condições iniciais e o resíduo da EDP.
    """
    def __init__(self, x_min, x_max, t_min, t_max, a, b, A=1.0):
        self.domain = {
            'x_min': torch.tensor(x_min),
            'x_max': torch.tensor(x_max),
            't_min': torch.tensor(t_min),
            't_max': torch.tensor(t_max)
        }
        self.a = a
        self.b = b
        self.A = A

    def ic(self, x):
        """ Condição inicial: eta(x,0) = A*sech^2(x), u(x,0) = 0 """
        eta_0 = self.A / torch.cosh(x)**2
        u_0 = torch.zeros_like(x)
        return eta_0, u_0

    def residual(self, model, x, t):
        """
        Calcula o resíduo das equações de Boussinesq.
        """
        x.requires_grad_(True)
        t.requires_grad_(True)

        eta, u = model(x, t)

        # Derivadas de primeira ordem
        eta_t = torch.autograd.grad(eta, t, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Termo não-linear (eta*u)_x
        eta_u_x = torch.autograd.grad(eta * u, x, grad_outputs=torch.ones_like(eta), create_graph=True)[0]

        # Derivadas para a segunda equação
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        eta_x = torch.autograd.grad(eta, x, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xxt = torch.autograd.grad(u_xx, t, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]

        # Equação 1: eta_t + u_x + a*(eta*u)_x = 0
        res_1 = eta_t + u_x + self.a * eta_u_x

        # Equação 2: u_t + eta_x + a*u*u_x - (b/3)*u_xxt = 0
        res_2 = u_t + eta_x + self.a * u * u_x - (self.b / 3.0) * u_xxt

        return res_1, res_2