import sys
sys.path.append('../functionall/')

import numpy as np
import toolbox
import random


np.random.seed(0)
random.seed(0)

# sample size
n = 100
# dimension d
d = 1
# error variance
sig = 0.4
# noise ratio
SNR = 3

# DGP: x and y
x = np.random.uniform(low=-1, high=1, size=(n, d))
y_true = np.linalg.norm(x, axis=1)**2

sigma = np.sqrt(np.var(y_true, ddof=1, axis=0)/SNR)
nse = np.random.normal(0, sigma, n)

y = (y_true + nse).reshape(n, 1)

# normalization
normalization = np.sqrt(np.sum(x**2, axis=0))/np.sqrt(x.shape[0])
x = x/normalization

# x = np.random.uniform(low=1, high=10, size=(n, d))
# nse = sig*np.random.normal(0, 1, n)
# # f = 3 + x[:,0]**0.5
# f = 3 + x[:,0]**0.2 + x[:,1]**0.3
# # f = 3 + x[:, 0]**0.05 + x[:, 1]**0.15 + x[:, 2]**0.3
# y = (f + nse).reshape(n, 1)

kfold = 5
u_para = np.linspace(0.001, 1, 50)

std, one = toolbox.L_opt(x, y, kfold, u_para)

print(std, one)