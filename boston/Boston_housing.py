import sys
sys.path.append('../functionall/')
import pandas as pd
import numpy as np
import random
import CNLS, CSVR, toolbox, LCR
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

random.seed(0)

# Calculate yhat in testing sample
def yhat(alpha, beta, x_test):

    # compute yhat for each testing observation
    yhat = np.zeros((len(x_test),))
    for i in range(len(x_test)):
        yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).min(axis=0)

    return yhat

# CSVR
def csvr_mse(x, y, i_mix):

    # u = np.linspace(0, 10, 20)
    # epsilon = np.array([0, 0.001, 0.01, 0.1, 0.2])
    # e_grid, u_grid = toolbox.GridSearch(x, y, kfold, epsilon=epsilon, u=u)
	
	
    error_out, error_in = [], []
    for k in range(kfold):
        # print("Fold", k, "\n")

        # divide up i.mix into K equal size chunks
        m = len(y) // kfold
        i_kfold = [i_mix[i:i+m] for i in range(0, len(i_mix), m)]
        if len(i_kfold) > kfold:
            i_kfold[-2:] = [i_kfold[-2]+i_kfold[-1]]
        	
        i_tr = toolbox.index_tr(k, i_kfold)
        i_val = i_kfold[k]

        # training predictors, training responses
        x_tr = x[i_tr, :]  
        y_tr = y[i_tr]   

        # validation predictors, validation responses
        x_val = x[i_val, :]
        y_val = y[i_val] 

        alpha, beta, ksia, ksib = CSVR.CSVR(y_tr, x_tr, epsilon=0.2, u=1.58)

        error_out.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))
        error_in.append(np.mean((yhat(alpha, beta, x_tr) - y_tr)**2))
    	
    mse = np.array([np.mean(np.array(error_out), axis=0), np.mean(np.array(error_in), axis=0)])
    std = np.array([np.std(np.array(error_out), axis=0), np.std(np.array(error_in), axis=0)])

    return mse, std

# SVR
def svr_mse(x, y, i_mix):

    error_out, error_in = [], []
    for k in range(kfold):
        # print("Fold", k, "\n")

        # divide up i.mix into K equal size chunks
        m = len(y) // kfold
        i_kfold = [i_mix[i:i+m] for i in range(0, len(i_mix), m)]
        if len(i_kfold) > kfold:
            i_kfold[-2:] = [i_kfold[-2]+i_kfold[-1]]
        
        i_tr = toolbox.index_tr(k, i_kfold)
        i_val = i_kfold[k]

        # training predictors, training responses
        x_tr = x[i_tr, :]  
        y_tr = y[i_tr]   

        # validation predictors, validation responses
        x_val = x[i_val, :]
        y_val = y[i_val] 

        para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
        svr = GridSearchCV(SVR(),para_grid)
        svr.fit(x_tr, y_tr)

        error_out.append(np.mean((svr.predict(x_val) - y_val)**2))
        error_in.append(np.mean((svr.predict(x_tr) - y_tr)**2))

    mse = np.array([np.mean(np.array(error_out), axis=0), np.mean(np.array(error_in), axis=0)])
    std = np.array([np.std(np.array(error_out), axis=0), np.std(np.array(error_in), axis=0)])

    return mse, std

# CNLS
def cnls_mse(x, y, i_mix):

    error_out, error_in = [], []
    for k in range(kfold):
        # print("Fold", k, "\n")

        # divide up i.mix into K equal size chunks
        m = len(y) // kfold
        i_kfold = [i_mix[i:i+m] for i in range(0, len(i_mix), m)]
        if len(i_kfold) > kfold:
            i_kfold[-2:] = [i_kfold[-2]+i_kfold[-1]]		

        i_tr = toolbox.index_tr(k, i_kfold)
        i_val = i_kfold[k]

        # training predictors, training responses
        x_tr = x[i_tr, :]  
        y_tr = y[i_tr]   

        # validation predictors, validation responses
        x_val = x[i_val, :]
        y_val = y[i_val] 

        model = CNLS.CNLS(y_tr, x_tr, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
        model.optimize(OPT_LOCAL)
        alpha, beta = model.get_alpha(), model.get_beta()

        error_out.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))
        error_in.append(np.mean((yhat(alpha, beta, x_tr) - y_tr)**2))

    mse = np.array([np.mean(np.array(error_out), axis=0), np.mean(np.array(error_in), axis=0)])
    std = np.array([np.std(np.array(error_out), axis=0), np.std(np.array(error_in), axis=0)])

    return mse, std


# convex regression
def lcr(x, y, i_mix):

    # l = np.linspace(0, 10, 20)
    # l_one = toolbox.L_opt(x, y, kfold, l)[1]

    error_out, error_in = [], []
    for k in range(kfold):
        # print("Fold", k, "\n")

        # divide up i.mix into K equal size chunks
        m = len(y) // kfold
        i_kfold = [i_mix[i:i+m] for i in range(0, len(i_mix), m)]
        if len(i_kfold) > kfold:
            i_kfold[-2:] = [i_kfold[-2]+i_kfold[-1]]

        i_tr = toolbox.index_tr(k, i_kfold)
        i_val = i_kfold[k]

        # training predictors, training responses
        x_tr = x[i_tr, :]  
        y_tr = y[i_tr]   

        # validation predictors, validation responses
        x_val = x[i_val, :]
        y_val = y[i_val] 

        alpha, beta, epsilon = LCR.LCR(y_tr, x_tr, L= 3.68)

        error_out.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))
        error_in.append(np.mean((yhat(alpha, beta, x_tr) - y_tr)**2))

    mse = np.array([np.mean(np.array(error_out), axis=0), np.mean(np.array(error_in), axis=0)])
    std = np.array([np.std(np.array(error_out), axis=0), np.std(np.array(error_in), axis=0)])

    return mse, std


# load data
data = pd.read_csv('Boston.csv')

# x = data.loc[:, ['NOX', 'DIS', 'PTRATIO', 'INDUS', 'TAX', 'ZN', 'RAD']]
x = data.loc[:, ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = data['MEDV']

x = np.array(x)
y = np.array(y)

kfold = 5
i_mix = random.sample(range(len(y)), k=len(y))

mse_csvr, std_csvr = csvr_mse(x, y, i_mix)
mse_cnls,std_cnls = cnls_mse(x, y, i_mix)
mse_svr, std_svr = svr_mse(x, y, i_mix)
mse_lcr, std_lcr = lcr(x, y, i_mix)
data = np.array([mse_csvr, std_csvr, mse_svr, std_svr, mse_cnls, std_cnls, mse_lcr, std_lcr]).T

df = pd.DataFrame(data, columns = ['csvr_mse', 'csvr_std', 'svr_mse', 'svr_std', 'cnls_mse', 'cnls_std', 'lcr_mse', 'lcr_std'])
df.to_excel('mse.xlsx')
#42.3754272930659 63.99884850752447 2327.985041400418 49.52787682333644 0.2 1.5789473684210527 3.6842105263157894