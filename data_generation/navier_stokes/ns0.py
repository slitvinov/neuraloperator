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
        ky = np.array([list(range(k_max)) + list(range(-k_max, 0))] * size)
        kx = ky.T
        self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * (
            (4 * (math.pi**2) * (kx**2 + ky**2) + tau**2)**(-alpha / 2.0))
        self.sqrt_eig[0, 0] = 0.0

    def sample(self, N):
        coeff = np.random.randn(N, self.size, self.size)
        coeff = self.sqrt_eig * coeff
        return np.fft.ifftn(coeff, axes=(-1, -2)).real


def navier_stokes_2d(w0, f, visc, T, delta_t, record_steps):
    N = np.shape(w0)[-1]
    print("N = ", N)
    k_max = math.floor(N / 2.0)
    steps = math.ceil(T / delta_t)
    w_h = np.fft.rfft2(w0)
    f_h = np.fft.rfft2(f)
    print(f_h.ndim)
    if f_h.ndim < w_h.ndim:
        f_h = np.expand_dims(f_h, 0)
    record_time = math.floor(steps / record_steps)
    ky = np.array([list(range(k_max)) + list(range(-k_max, 0))] * N)
    kx = ky.T
    kx = kx[..., :k_max + 1]
    ky = ky[..., :k_max + 1]
    lap = 4 * (math.pi**2) * (kx**2 + ky**2)
    lap[0, 0] = 1.0
    dealias = np.expand_dims(
        np.logical_and(
            np.abs(ky) <= (2.0 / 3.0) * k_max,
            np.abs(kx) <= (2.0 / 3.0) * k_max), 0)
    sol = np.zeros((*w0.shape, record_steps))
    sol_t = np.zeros(record_steps)
    c = 0
    t = 0.0
    for j in range(steps):
        if j % 1000 == 0:
            print(j, steps)
        psi_h = w_h / lap
        q = 2. * math.pi * ky * 1j * psi_h
        q = np.fft.irfft2(q, s=(N, N))
        v = -2. * math.pi * kx * 1j * psi_h
        v = np.fft.irfft2(v, s=(N, N))
        w_x = 2. * math.pi * kx * 1j * w_h
        w_x = np.fft.irfft2(w_x, s=(N, N))
        w_y = 2. * math.pi * ky * 1j * w_h
        w_y = np.fft.irfft2(w_y, s=(N, N))
        F_h = np.fft.rfft2(q * w_x + v * w_y)
        F_h = dealias * F_h
        w_h = (-delta_t * F_h + delta_t * f_h +
               (1.0 - 0.5 * delta_t * visc * lap) * w_h) / (
                   1.0 + 0.5 * delta_t * visc * lap)
        t += delta_t
        if (j + 1) % record_time == 0:
            w = np.fft.irfft2(w_h, s=(N, N))
            sol[..., c] = w
            sol_t[c] = t
            c += 1
    return sol, sol_t


# np.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)
# s = 256
# T = 50
s = 32
T = 0.5

N = 20
GRF = GaussianRF(s, alpha=2.5, tau=7)
#Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = np.linspace(0, 1, s + 1)
t = t[0:-1]
X, Y = np.meshgrid(t, t, indexing='ij')
f = 0.1 * (np.sin(2 * math.pi * (X + Y)) + np.cos(2 * math.pi * (X + Y)))
record_steps = 200
a = np.zeros((N, s, s))
u = np.zeros((N, s, s, record_steps))
bsize = 20
c = 0
for j in range(N // bsize):
    w0 = GRF.sample(bsize)
    sol, sol_t = navier_stokes_2d(w0, f, 1e-3, T, 1e-4, record_steps)
    a[c:(c + bsize), ...] = w0
    u[c:(c + bsize), ...] = sol
    c += bsize
    print("jc: ", j, c, w0.shape, sol.shape)
scipy.io.savemat('ns_data.mat',
                 mdict={
                     'a': a,
                     'u': u,
                     't': sol_t,
                 })
