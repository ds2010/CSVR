import numpy as np
import random
import LCR
import toolbox

np.random.seed(0)
random.seed(0)

# sample size
n = 50
# dimension d
d = 2
# error variance
sig = 0.5

# DGP: x and y
x = np.random.uniform(low=1, high=10, size=(n, d))
nse = sig*np.random.normal(0, 1.2, n)
y_true = 3 + x[:,0]**0.2 + x[:,1]**0.3
y = (y_true + nse).reshape(n, 1)


kfold = 5
L_para = np.linspace(0.1, 10, 50)

std, one = toolbox.L_opt(x, y, kfold, L_para)

# alpha, beta, epsilon = LCR.LCR(y, x, 2)

# print(epsilon)


