import numpy as np
import random
import MC

np.random.seed(0)
random.seed(0)

M=10
n=100
d=3
epsilon = 0.4

MSE_all = np.zeros((M, 5))
for i in range(M):
    MSE_all[i, :] = MC.simulation(n, d, epsilon)

MSE = np.mean(MSE_all, axis=0)

print(MSE)
