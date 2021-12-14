import numpy as np
import random
import MC

np.random.seed(0)
random.seed(0)

M=10
n=50
d=3
SNR=3

MSE_all = np.zeros((M, 1))
for i in range(M):
    MSE_all[i, :] = MC.simulation(n, d, SNR)

MSE = np.mean(MSE_all, axis=0)

print(MSE_all)
