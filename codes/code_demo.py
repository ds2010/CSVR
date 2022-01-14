import sys
sys.path.append('../functionall')
import numpy as np
import pandas as pd
import random
import DGP
import toolbox
import MC

def main():

    # cross validation
    e_all, u_all, re_all, x_all, y_all, y_true_all = [], [], [], [], [], []
    for i in range(M):
        kfold = 2
        u = np.array([0.1, 0.5, 1, 2, 5])
        epsilon = np.array([0, 0.001, 0.01, 0.1, 0.2])

        # DGP
        x, y, y_true = DGP.inputs(n, d, sig)
        # print(y)

        # e_grid, u_grid = toolbox.GridSearch(x, y, kfold, epsilon=epsilon, u=u)
        # e_all.append(e_grid)
        # u_all.append(u_grid)
        x_all.append(x)
        y_all.append(y)
        y_true_all.append(y_true)
    # e_para, u_para = np.mean(np.array(e_all)), np.mean(np.array(u_all))
    # e_std, u_std = np.std(np.array(e_all)), np.std(np.array(u_all))
    x, y, y_true = np.array(x_all), np.array(y_all), np.array(y_true_all)

    #simulations
    re_all = []
    for i in range(M):
        re_all.append(MC.simulation(x[i], y[i], y_true[i], 0.09868, 2.53))

    data = np.array([np.mean(np.array(re_all), axis=0), 
                        np.std(np.array(re_all), axis=0)])

    df = pd.DataFrame(data, columns = ['csvr', 'svr', 'cnls'])
    print(df)

if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    M=50
    n=50
    d=1
    sig = 0.5

    main()