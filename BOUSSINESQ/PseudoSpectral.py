import torch

class PseudoSpectralBoussinesq:
    def __init__(self, boussinesq, Nx=256, Nt=1000, device='cpu'):
        self.Nx = Nx # discretizacao do dominio fisico (precisa ser 2^s para usar fft)
        self.Nt = Nt # discretizacao do dominio temporal (é simples pois usamos RK4 apenas)
        self.device = device
        self.a = boussinesq.a
        self.b = boussinesq.b

        self.t = [0.0]
        self.dt = (boussinesq.domain['t_max'] - boussinesq.domain['t_min']) / Nt

        self.x_min = boussinesq.domain['x_min']
        self.x_max = boussinesq.domain['x_max']
        self.x = torch.linspace(self.x_min.item(), self.x_max.item(), Nx, device=device)
        self.dx = self.x[1] - self.x[0]

        self.k = 2 * torch.pi * torch.fft.fftfreq(Nx, d=self.dx.item()).to(device)
        self.ik = 1j * self.k
        self.k2 = self.k ** 2
        self.k3 = self.k ** 3

        eta0, u0 = boussinesq.ic(self.x)
        self.eta_hat = torch.fft.fft(eta0)
        self.u_hat = torch.fft.fft(u0)

        # salva apenas os campos no domínio físico
        eta = torch.fft.ifft(self.eta_hat).real
        u = torch.fft.ifft(self.u_hat).real
        self.history = [(eta.clone(), u.clone())]

    # eta_hat_t, u_hat_t = field(eta_hat, u_hat)
    def field(self, eta_hat, u_hat):
        # precisamos recuperar eta e u para calcular termos não-lineares
        eta = torch.fft.ifft(eta_hat).real
        u = torch.fft.ifft(u_hat).real
        eta_x = torch.fft.ifft(self.ik * eta_hat).real
        u_x = torch.fft.ifft(self.ik * u_hat).real

        # fft(eta_x * u + eta * u_x)
        nonlinear_eta_hat = torch.fft.fft(eta_x * u + eta * u_x)

        # fft(0.5 * u^2)
        nonlinear_u_hat = self.ik * torch.fft.fft(0.5 * u ** 2)

        # fft(-(b/3) * u_xxt)
        u_xxt_hat = self.ik * (-self.k2 * u_hat)
        dissipative_term = -(self.b / 3) * u_xxt_hat

        # EDP pronta para fazer o passo temporal
        eta_t_hat = -self.ik * u_hat - self.a * nonlinear_eta_hat

        numerator = -self.ik * eta_hat - self.a * nonlinear_u_hat
        denominator = 1 + (self.b / 3) * self.k2
        u_t_hat = numerator / denominator

        return eta_t_hat, u_t_hat

    def RK4_step(self, eta_hat, u_hat):
        dt = self.dt

        # calculando pesos do Runge-Kutta
        k1_eta, k1_u = self.field(eta_hat, u_hat)
        k2_eta, k2_u = self.field(eta_hat + 0.5 * dt * k1_eta, u_hat + 0.5 * dt * k1_u)
        k3_eta, k3_u = self.field(eta_hat + 0.5 * dt * k2_eta, u_hat + 0.5 * dt * k2_u)
        k4_eta, k4_u = self.field(eta_hat + dt * k3_eta, u_hat + dt * k3_u)

        eta_hat_new = eta_hat + (dt / 6) * (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta)
        u_hat_new = u_hat + (dt / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)

        return eta_hat_new, u_hat_new # retorna o próximo campo

    def solve(self, save_every=10):
        eta_hat, u_hat = self.eta_hat.clone(), self.u_hat.clone()  # tem que clonar pois senão salvaria apenas o endereço da memória
        
        for n in range(1, self.Nt + 1):
            eta_hat, u_hat = self.RK4_step(eta_hat, u_hat)

            if n % save_every == 0:
                # salva diretamente os campos físicos
                eta = torch.fft.ifft(eta_hat).real
                u = torch.fft.ifft(u_hat).real

                self.t.append(n * self.dt)
                self.history.append((eta.clone(), u.clone()))

                print(f"Saved at step {n}")

        return self.get_solution_history()

    def get_solution_history(self):
        # separa os valores de eta e u manualmente
        etas = []
        us = []
        for eta, u in self.history:
            etas.append(eta)
            us.append(u)
        return self.x, self.t, etas, us
    