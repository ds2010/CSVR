import numpy as np
import random
import LCR

np.random.seed(0)
random.seed(0)

# sample size
n = 20
# dimension d
d = 2
# error variance
sig = 0.5

# DGP: x and y
x = np.random.uniform(low=1, high=10, size=(n, d))
nse = sig*np.random.normal(0, 1.2, n)
y_true = 3 + x[:,0]**0.2 + x[:,1]**0.3
y = y_true + nse

L = 2

alpha, beta, epsilon = LCR.LCR(y, x, L)

print(epsilon)


