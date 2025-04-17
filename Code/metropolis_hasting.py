import numpy as np
import random

def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta # Unpack the parameters
    model = m*x + b
    sigma2 = yerr**2 + model**2*np.exp(2*log_f)
    return -0.5*np.sum((y - model)**2/sigma2 + np.log(sigma2))

m_true = -0.9594
b_true = 4.294
f_true = 0.534

np.random.seed(42)
p_0 = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
x, y, yerr = np.loadtxt("data.txt", unpack=True)

accepted = []
trial = []

N_steps = 10000
N_move = 0

L_0 = np.exp(log_likelihood(p_0, x, y, yerr))
print(L_0)
accepted.append(p_0)
trial.append(p_0)
for i in range(0, N_steps):
    p = np.array([p_0[0], p_0[1], p_0[2]]) + 0.05*np.array([p_0[0], p_0[1], p_0[2]])*np.random.randn(3)
    L_i = np.exp(log_likelihood(p, x, y, yerr))
    ratio = L_i/L_0
    trial.append(p)
    r = random.random()
    print(L_i, ratio, r)
    if (ratio > r):
        accepted.append(p)
        p_0 = p
        L_0 = L_i
        N_move += 1
    else:
        accepted.append(p_0)

np.savetxt("accepted.dat", accepted)
np.savetxt("trial.dat", trial)
print (100.0*N_move/N_steps)
