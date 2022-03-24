import numpy as np
import pandas as pd
import random
import DGP
import toolbox
import MC


def main():

    # cross validation
    e_all, u_all, l_all = [], [], []
    re_all, x_all, y_all, y_true_all = [], [], [], []

    for i in range(M):
        kfold = 5
        u = np.array([0.1, 0.5, 1, 2, 5])
        epsilon = np.array([0, 0.001, 0.01, 0.1, 0.2])
        l = np.array([0.1, 0.5, 1, 2, 5])

        # DGP
        x, y, y_true = DGP.inputs(n, d, sig)
        # CSVR tuning
        e_grid, u_grid = toolbox.GridSearch(x, y, kfold, epsilon=epsilon, u=u)
        # LCR tuning
        l_one = toolbox.L_opt(x, y, kfold, l)[1]

        e_all.append(e_grid)
        u_all.append(u_grid)
        l_all.append(l_one)

        x_all.append(x)
        y_all.append(y)
        y_true_all.append(y_true)

    # parameters by Monte Carlo
    e_para, u_para, l_one = np.mean(np.array(e_all)), np.mean(np.array(u_all)), np.mean(np.array(l_all))
    e_std, u_std = np.std(np.array(e_all)), np.std(np.array(u_all))
    x, y, y_true = np.array(x_all), np.array(y_all), np.array(y_true_all)
    
    #simulations
    for i in range(M):
        re_all.append(MC.simulation(x[i], y[i], y_true[i], e_para, u_para, l_one))

    data = np.array([np.append(np.mean(np.array(re_all), axis=0),[e_para,u_para]), 
                        np.append(np.std(np.array(re_all), axis=0),[e_std, u_std])])

    df = pd.DataFrame(data, columns = ['csvr', 'svr', 'cnls', 'lcr', 'e_grid', 'u_grid'])
    df.to_csv('code' + '{0}_{1}_{2}.csv'.format(n, d, sig))


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    M=50
    n=50
    d=1
    sig = 0.5

    main()