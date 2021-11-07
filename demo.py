import numpy as np
import cross_validation


np.random.seed(0)

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

n_fold = 5
u_parameter = [10**i for i in np.linspace(-3, 1, 50)]

u = cross_validation.stand_error(X, y, n_fold, u_parameter)

print(u)

