import numpy as np
import random
import CNLS, CSVR, DGP, toolbox
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def simulation(n, d, sig):
    
    kfold = 10
    para = np.linspace(0.01, 5, 50)
    # epsilon = np.array([0, 0.001, 0.01, 0.1, 0.2])

    # DGP
    x, y, y_true = DGP.inputs(n, d, sig)

    # solve the CSVR model
    # u_grid = toolbox.u_opt(x, y, kfold, epsilon=0.1, u_para=para)
    alpha, beta, ksia, ksib = CSVR.CSVR(y, x, epsilon=0.1052, u=3.0145)
    y_csvr = alpha + np.sum(beta * x, axis=1)
    mse_csvr = np.mean((y_true - y_csvr)**2)

    # solve the SVR model
    para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
    svr = GridSearchCV(SVR(),para_grid)
    svr.fit(x, y)
    y_svr = svr.predict(x)
    mse_svr = np.mean((y_true - y_svr)**2)

    # solve the CNLS model
    model1 = CNLS.CNLS(y, x, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
    model1.optimize(OPT_LOCAL)
    mse_cnls = np.mean((model1.get_frontier() - y_true)**2)


    return mse_csvr, mse_svr, mse_cnls

if __name__ == '__main__':


    np.random.seed(0)
    random.seed(0)

    M=50
    n=100
    d=1
    sig = 0.5

    re_all = []
    for i in range(M):
        re_all.append(simulation(n, d, sig))
    da_all = np.array(re_all)
    data = np.array([np.mean(da_all, axis=0), np.std(da_all, axis=0)])

    print(data)
    # 0.01796609 0.03002438 0.02193116
    # 0.01805379 0.03002438 0.02193116
