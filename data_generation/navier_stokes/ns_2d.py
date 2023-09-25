import torch
import math
import scipy.io
import numpy as np
import random


class GaussianRF:

    def __init__(self, size, alpha, tau):
        self.dim = 2
        self.size = size
        sigma = tau**(0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        ky = torch.tensor([list(range(k_max)) + list(range(-k_max, 0))] * size)
        kx = ky.T
        self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * (
            (4 * (math.pi**2) * (kx**2 + ky**2) + tau**2)**(-alpha / 2.0))
        self.sqrt_eig[0, 0] = 0.0

    def sample(self, N):
        coeff = torch.randn(N, self.size, self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        return torch.fft.ifftn(coeff, dim=(-1, -2)).real


torch.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)
# s = 256
# T = 50
s = 32
T = 0.5

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
w0 = GRF.sample(bsize)
N = w0.size()[-1]
visc = 1e-3
delta_t = 1e-4
k_max = math.floor(N / 2.0)
steps = math.ceil(T / delta_t)
w_h = torch.fft.rfft2(w0)
f_h = torch.fft.rfft2(f)
if len(f_h.size()) < len(w_h.size()):
    f_h = torch.unsqueeze(f_h, 0)
record_time = math.floor(steps / record_steps)
ky = torch.tensor([list(range(k_max)) + list(range(-k_max, 0))] * N)
kx = ky.T
kx = kx[..., :k_max + 1]
ky = ky[..., :k_max + 1]
lap = 4 * (math.pi**2) * (kx**2 + ky**2)
lap[0, 0] = 1.0
dealias = torch.unsqueeze(
    torch.logical_and(
        torch.abs(ky) <= (2.0 / 3.0) * k_max,
        torch.abs(kx) <= (2.0 / 3.0) * k_max).float(), 0)
sol = torch.zeros(*w0.size(), record_steps)
sol_t = torch.zeros(record_steps)
c = 0
t = 0.0
for j in range(steps):
    if j % 1000 == 0:
        print(j, steps)
    psi_h = w_h / lap
    q = 2. * math.pi * ky * 1j * psi_h
    q = torch.fft.irfft2(q, s=(N, N))
    v = -2. * math.pi * kx * 1j * psi_h
    v = torch.fft.irfft2(v, s=(N, N))
    w_x = 2. * math.pi * kx * 1j * w_h
    w_x = torch.fft.irfft2(w_x, s=(N, N))
    w_y = 2. * math.pi * ky * 1j * w_h
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
scipy.io.savemat('ns_data.mat',
                 mdict={
                     'a': w0.cpu().numpy(),
                     'u': sol.cpu().numpy(),
                     't': sol_t.cpu().numpy()
                 })
