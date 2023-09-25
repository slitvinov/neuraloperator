import torch
import math
import scipy.io


class GaussianRF:
    def __init__(self, size, alpha, tau):
        self.dim = 2
        sigma = tau**(0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                                torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size,1)
        k_x = wavenumers.transpose(0, 1)
        k_y = wavenumers
        self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * (
            (4 * (math.pi**2) * (k_x**2 + k_y**2) + tau**2)**(-alpha / 2.0))
        self.sqrt_eig[0, 0] = 0.0
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self, N):
        coeff = torch.randn(N, *self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1,
                                                     -1))).real


def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    N = w0.size()[-1]
    k_max = math.floor(N / 2.0)
    steps = math.ceil(T / delta_t)
    w_h = torch.fft.rfft2(w0)
    f_h = torch.fft.rfft2(f)
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)
    record_time = math.floor(steps / record_steps)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1),
                     torch.arange(start=-k_max, end=0, step=1)),
                    0).repeat(N, 1)
    k_x = k_y.transpose(0, 1)
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]
    lap = 4 * (math.pi**2) * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0
    dealias = torch.unsqueeze(
        torch.logical_and(
            torch.abs(k_y) <= (2.0 / 3.0) * k_max,
            torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)
    sol = torch.zeros(*w0.size(), record_steps)
    sol_t = torch.zeros(record_steps)
    c = 0
    t = 0.0
    for j in range(steps):
        print(j, steps)
        psi_h = w_h / lap
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))
        F_h = torch.fft.rfft2(q * w_x + v * w_y)
        F_h = dealias * F_h
        w_h = (-delta_t * F_h + delta_t * f_h +
               (1.0 - 0.5 * delta_t * visc * lap) * w_h) / (
                   1.0 + 0.5 * delta_t * visc * lap)
        t += delta_t
        if (j + 1) % record_time == 0:
            w = torch.fft.irfft2(w_h, s=(N, N))
            sol[..., c] = w
            sol_t[c] = t
            c += 1
    return sol, sol_t


# s = 256
s = 32
N = 20
GRF = GaussianRF(s, alpha=2.5, tau=7)
#Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s + 1)
t = t[0:-1]
X, Y = torch.meshgrid(t, t, indexing='ij')
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
record_steps = 200
a = torch.zeros(N, s, s)
u = torch.zeros(N, s, s, record_steps)
bsize = 20
c = 0
for j in range(N // bsize):
    print(j)
    w0 = GRF.sample(bsize)
    sol, sol_t = navier_stokes_2d(w0, f, 1e-3, 50.0, 1e-4, record_steps)
    a[c:(c + bsize), ...] = w0
    u[c:(c + bsize), ...] = sol
    c += bsize
    print(j, c)
scipy.io.savemat('ns_data.mat',
                 mdict={
                     'a': a.cpu().numpy(),
                     'u': u.cpu().numpy(),
                     't': sol_t.cpu().numpy()
                 })
