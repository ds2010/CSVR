import numpy as np
import toolbox
import random


np.random.seed(0)
random.seed(0)

# sample size
n = 100
# dimension d
d = 3
# error variance
sig = 0.4

# DGP: x and y
x = np.random.uniform(low=1, high=10, size=(n, d))
nse = sig*np.random.normal(0, 1, n)
# f = 3 + x[:,0]**0.5
# f = 3 + x[:,0]**0.2 + x[:,1]**0.3
f = 3 + x[:, 0]**0.05 + x[:, 1]**0.15 + x[:, 2]**0.3
y = (f + nse).reshape(n, 1)

kfold = 5
u_para = np.linspace(0.001, 1, 50)

std, one = toolbox.e_opt(x, y, kfold, u_para)

print(std, one)