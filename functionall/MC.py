import numpy as np
import CNLS, CSVR, LCR, DGP, toolbox
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# Calculate yhat in testing sample
def yhat(alpha, beta, x_test):
    '''
    function estimate the y_hat of convex functions.
    refers to equation (4.1) in journal article:
    "Representation theorem for convex nonparametric least squares. Timo Kuosmanen (2008)"
    input:
    alpha and beta are regression coefficients; x_test is the input of test sample.
    output:
    return the estimated y_hat.
    '''

    # compute yhat for each testing observation
    yhat = np.zeros((len(x_test),))
    for i in range(len(x_test)):
        yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).min(axis=0)

    return yhat

def simulation(n, d, sig):
    '''
    function of in-sample simulation.
    input:
    n, d, sig are parameters of DGP.
    output:
    return four MSEs of four methods.
    '''
    # generate input data
    x, y, y_true = DGP.inputs(n, d, sig)

    # tuning parameters
    kfold = 5
    epsilon, para = np.array([0.001, 0.01, 0.1, 0.2, 0.5]), np.linspace(0.1, 8, 11)
    L_para = np.array([0.1, 0.5, 1, 2, 5])
    e = toolbox.GridSearch(x, y, kfold, epsilon, para)[0]
    u = toolbox.GridSearch(x, y, kfold, epsilon, para)[1]
    l = toolbox.L_opt(x, y, kfold, L_para)[1]
    
    # solve the CSVR model
    alpha, beta, ksia, ksib = CSVR.CSVR(y, x, e, u)
    y_csvr = alpha + np.sum(beta * x, axis=1)
    mse_csvr = np.mean((y_true - y_csvr)**2)
    mae_csvr = np.mean(np.absolute(y_true - y_csvr))

    # solve the RBF SVR model
    para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
    svr = GridSearchCV(SVR(kernel='rbf'),para_grid)
    svr.fit(x, y)
    y_svr = svr.predict(x)
    mse_svr = np.mean((y_true - y_svr)**2)
    mae_svr = np.mean(np.absolute(y_true - y_svr))

    # solve the polynomial SVR model
    para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
    svrp = GridSearchCV(SVR(kernel='poly'),para_grid)
    svrp.fit(x, y)
    y_svrp = svrp.predict(x)
    mse_svrp = np.mean((y_true - y_svrp)**2)
    mae_svrp = np.mean(np.absolute(y_true - y_svrp))

    # solve the CNLS model
    model1 = CNLS.CNLS(y, x, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
    model1.optimize(OPT_LOCAL)
    mse_cnls = np.mean((model1.get_frontier() - y_true)**2)
    mae_cnls = np.mean(np.absolute(y_true - model1.get_frontier()))

    # solve the LCR model
    alpha, beta, epsilon = LCR.LCR(y, x, l)
    y_lcr = alpha + np.sum(beta * x, axis=1)
    mse_lcr = np.mean((y_true - y_lcr)**2)
    mae_lcr = np.mean(np.absolute(y_true - y_lcr))

    return mse_csvr, mse_svr, mse_svrp, mse_cnls, mse_lcr, mae_csvr, mae_svr, mae_svrp, mae_cnls, mae_lcr


def simulation_out(n, d, sig, nt):
    '''
    function of out-of-sample simulation.
    input:
    n, d, sig are parameters of DGP. e, u are tuning parameters(epsilon and C) of CSVR; 
    l is the tuning parameter of LCR. 
    nt indecates the amount of out-of-sample data.
    output:
    return four MSEs of four methods.
    '''

    # generate train and test sample
    x, y, y_true = DGP.inputs(n+nt, d, sig)
    x_tr, y_tr = x[:n,:], y[:n]
    x_te, y_te = x[-nt:,:], y_true[-nt:]

    # tuning parameters
    kfold = 5
    epsilon, para = np.array([0.001, 0.01, 0.1, 0.2, 0.5]), np.linspace(0.1, 8, 11)
    L_para = np.array([0.1, 0.5, 1, 2, 5])
    e = toolbox.GridSearch(x_tr, y_tr, kfold, epsilon, para)[0]
    u = toolbox.GridSearch(x_tr, y_tr, kfold, epsilon, para)[1]
    l = toolbox.L_opt(x_tr, y_tr, kfold, L_para)[1]

    # solve the CSVR model
    alpha, beta, ksia, ksib = CSVR.CSVR(y_tr, x_tr, e, u)
    y_csvr = yhat(alpha, beta, x_te)
    mse_csvr = np.mean((y_te - y_csvr)**2)
    mae_csvr = np.mean(np.absolute(y_te - y_csvr))

    # solve the RBF SVR model
    para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
    svr = GridSearchCV(SVR(kernel='rbf'),para_grid)
    svr.fit(x_tr, y_tr)
    y_svr = svr.predict(x_te)
    mse_svr = np.mean((y_te - y_svr)**2)
    mae_svr = np.mean(np.absolute(y_te - y_svr))

    # solve the polynomial SVR model
    para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
    svrp = GridSearchCV(SVR(kernel='poly'),para_grid)
    svrp.fit(x_tr, y_tr)
    y_svrp = svrp.predict(x_te)
    mse_svrp = np.mean((y_te - y_svrp)**2)
    mae_svrp = np.mean(np.absolute(y_te - y_svrp))

    # solve the CNLS model
    model = CNLS.CNLS(y_tr, x_tr, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
    model.optimize(OPT_LOCAL)
    alpha, beta = model.get_alpha(), model.get_beta()
    y_cnls = yhat(alpha, beta, x_te)
    mse_cnls = np.mean((y_te - y_cnls)**2)
    mae_cnls = np.mean(np.absolute(y_te - y_cnls))

    # solve the LCR model
    alpha, beta, epsilon = LCR.LCR(y_tr, x_tr, l)
    y_lcr = yhat(alpha, beta, x_te)
    mse_lcr = np.mean((y_te - y_lcr)**2)
    mae_lcr = np.mean(np.absolute(y_te - y_lcr))

    return mse_csvr, mse_svr, mse_svrp, mse_cnls, mse_lcr, mae_csvr, mae_svr, mae_svrp, mae_cnls, mae_lcr

def simulation_outlier(n, d, sig, nt):
    '''
    function of in-sample simulation with outliers.
    input:
    n, d, sig are parameters of DGP. nt is the number of outliers.
    output:
    return four MSEs of four methods.
    '''

    x, y, y_true = DGP.outlier(n, d, sig, nt)
    kfold = 5
    epsilon, para = np.array([0.001, 0.01, 0.1, 0.2, 0.5]), np.linspace(0.1, 8, 11)
    L_para = np.array([0.1, 0.5, 1, 2, 5])
    e = toolbox.GridSearch(x, y, kfold, epsilon, para)[0]
    u = toolbox.GridSearch(x, y, kfold, epsilon, para)[1]
    l = toolbox.L_opt(x, y, kfold, L_para)[1]

    # solve the CSVR model
    alpha, beta, ksia, ksib = CSVR.CSVR(y, x, e, u)
    y_csvr = alpha + np.sum(beta * x, axis=1)
    mse_csvr = np.mean((y_true - y_csvr)**2)
    mae_csvr = np.mean(np.absolute(y_true - y_csvr))

    # solve the RBF SVR model
    para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
    svr = GridSearchCV(SVR(kernel='rbf'),para_grid)
    svr.fit(x, y)
    y_svr = svr.predict(x)
    mse_svr = np.mean((y_true - y_svr)**2)
    mae_svr = np.mean(np.absolute(y_true - y_svr))

    # solve the polynomial SVR model
    para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
    svrp = GridSearchCV(SVR(kernel='poly'),para_grid)
    svrp.fit(x, y)
    y_svrp = svrp.predict(x)
    mse_svrp = np.mean((y_true - y_svrp)**2)
    mae_svrp = np.mean(np.absolute(y_true - y_svrp))

    # solve the CNLS model
    model1 = CNLS.CNLS(y, x, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
    model1.optimize(OPT_LOCAL)
    mse_cnls = np.mean((model1.get_frontier() - y_true)**2)
    mae_cnls = np.mean(np.absolute(y_true - model1.get_frontier()))

    # solve the LCR model
    alpha, beta, epsilon = LCR.LCR(y, x, l)
    y_lcr = alpha + np.sum(beta * x, axis=1)
    mse_lcr = np.mean((y_true - y_lcr)**2)
    mae_lcr = np.mean(np.absolute(y_true - y_lcr))

    return mse_csvr, mse_svr, mse_svrp, mse_cnls, mse_lcr, mae_csvr, mae_svr, mae_svrp, mae_cnls, mae_lcr

def simulation_poly(n, d, sig):
    '''
    function of in-sample simulation.
    input:
    n, d, sig are parameters of DGP.
    output:
    return four MSEs of four methods.
    '''
    # generate train and test sample
    nt = 1000
    x, y, y_true = DGP.inputs(n+nt, d, sig)
    x_tr, y_tr = x[:n,:], y[:n]
    x_te, y_te = x[-nt:,:], y_true[-nt:]

    
    # solve the RBF SVR model
    para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
    svr = GridSearchCV(SVR(kernel='rbf'),para_grid)
    svr.fit(x_tr, y_tr)
    y_svr = svr.predict(x_te)
    mse_svr = np.mean((y_te - y_svr)**2)
    mape_svr = np.mean(np.absolute((y_te - y_svr)/y_te))*100

    # solve the polynomial SVR model
    para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
    svrp = GridSearchCV(SVR(kernel='poly'),para_grid)
    svrp.fit(x_tr, y_tr)
    y_svrp = svrp.predict(x_te)
    mse_svrp = np.mean((y_te - y_svrp)**2)
    mape_svrp = np.mean(np.absolute((y_te - y_svrp)/y_te))*100


    return mse_svr, mse_svrp, mape_svr, mape_svrp