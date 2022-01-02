import numpy as np
import pandas as pd
import random
import DGP
import toolbox
import MC


def main():

    # cross validation
    e_all, u_all, re_all = [], [], []
    for i in range(M):
        kfold = 10
        u = np.array([0.1, 0.5, 1, 2, 5])
        epsilon = np.array([0, 0.001, 0.01, 0.1, 0.2])

        # DGP
        x, y, y_true = DGP.inputs(n, d, sig)

        e_grid, u_grid = toolbox.GridSearch(x, y, kfold, epsilon=epsilon, u=u)
        e_all.append(e_grid)
        u_all.append(u_grid)
    e_para, u_para = np.mean(np.array(e_all)), np.mean(np.array(u_all))
    e_std, u_std = np.std(np.array(e_all)), np.std(np.array(u_all))

    #simulations
    for i in range(M):
        re_all.append(MC.simulation(n, d, sig, e_para, u_para))

    data = np.array([np.append(np.mean(np.array(re_all), axis=0),[e_para,u_para]), 
                        np.append(np.std(np.array(re_all), axis=0),[e_std, u_std])])

    df = pd.DataFrame(data, columns = ['csvr', 'svr', 'cnls', 'e_grid', 'u_grid'])
    df.to_csv('code' + '{0}_{1}_{2}.csv'.format(n, d, sig))


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    M=50
    n=200
    d=3
    sig = 1

    main()