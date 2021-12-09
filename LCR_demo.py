import numpy as np
import random
import LCR

np.random.seed(0)
random.seed(0)

# sample size
n = 100
# dimension d
d = 2
# error variance
sig = 0.5

# DGP: x and y
X = np.random.uniform(low=1, high=10, size=(n, d))
nse = sig*np.random.normal(0, 1, n)
f = np.linalg.norm(X, axis=1)**2
y = (f + nse).reshape(n, 1)

L = 10

alpha, beta, epsilon = LCR.LCR(y, X, L)

print(epsilon)

