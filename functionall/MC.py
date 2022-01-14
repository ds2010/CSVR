import numpy as np
import CNLS, CSVR
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def simulation(x, y, y_true, e_para, u_para):

    
    # solve the CSVR model
    alpha, beta, ksia, ksib = CSVR.CSVR(y, x, epsilon=e_para, u=u_para)
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
