# import sys
# sys.path.append('../functionall')
import numpy as np
import pandas as pd
import random
import CNLS, CSVR, DGP, toolbox
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def simulation(n, d, sig, e_para, u_para):
    

    # DGP
    x, y, y_true = DGP.inputs(n, d, sig)

    # solve the CSVR model
    # u_grid = toolbox.u_opt(x, y, kfold, epsilon=0.1, u_para=para)
    alpha, beta, ksia, ksib = CSVR.CSVR(y, x, epsilon=e_para, u=u_para)
    y_csvr = alpha + np.sum(beta * x, axis=1)
    mse_csvr = np.mean((y_true - y_csvr)**2)


    return mse_csvr

if __name__ == '__main__':


    np.random.seed(0)
    random.seed(0)

    M=50
    n=100
    d=3
    sig = 2

    para = pd.read_csv('code' + '{0}_{1}_{2}.csv'.format(n, d, sig), skipinitialspace=True, usecols=['e_grid', 'u_grid'], nrows=1)
    e_para, u_para = np.array(para)[0][0], np.array(para)[0][1]
    re_all = []
    for i in range(M):
        re_all.append(simulation(n, d, sig, e_para, u_para))
    da_all = np.array(re_all)
    data = np.array([np.mean(da_all, axis=0), np.std(da_all, axis=0)])

    print(data)
    # 0.01796609 0.03002438 0.02193116
    # 0.01805379 0.03002438 0.02193116
