import numpy as np
import random
import MC




def main():

    MSE_all = np.zeros((M, 5))
    for i in range(M):
        MSE_all[i, :] = MC.simulation(n, d, epsilon=e)

    MSE = np.mean(MSE_all, axis=0)

    np.savetxt('code' + '{0}_{1}_{2}.txt'.format(n, d, e), MSE, fmt = '%-.4f')

if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    M=50
    n=200
    d=3
    e=0.2

    main()