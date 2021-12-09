import numpy as np
import toolbox
import random


np.random.seed(0)
random.seed(0)

# sample size
n = 50
# dimension d
d = 2
# error variance
sig = 0.2

# DGP: x and y
X = np.random.uniform(low=1, high=10, size=(n, d))
nse = sig*np.random.normal(0, 1, n)
f = np.linalg.norm(X, axis=1)**2
y = (f + nse).reshape(n, 1)

kfold = 5
u_para = np.linspace(0.001, 2.5, 50)

u = toolbox.u_opt(X, y, kfold, u_para)


print(u)