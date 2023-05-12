import sys
sys.path.append('../functionall/')
import pandas as pd
import numpy as np
import random
import CNLS, CSVR, toolbox, LCR
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split


# Calculate yhat in testing sample
def yhat(alpha, beta, x_test):

    # compute yhat for each testing observation
    yhat = np.zeros((len(x_test),))
    for i in range(len(x_test)):
        yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).min(axis=0)

    return yhat


# CSVR
def csvr_mse(x, y, i_mix):

    u = [0.01, 0.1, 1, 2, 5, 10, 20]
    epsilon = [0.001, 0.01, 0.1, 0.2, 0.8]
    e_grid, u_grid = toolbox.GridSearch(x, y, kfold, epsilon=epsilon, u=u)
    # e_grid, u_grid = 0.1, 15
	
    mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in = [], [], [], [], [], [], [], []
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

        alpha, beta, ksia, ksib = CSVR.CSVR(y_tr, x_tr, e_grid, u_grid)

        mse_out.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))
        mse_in.append(np.mean((yhat(alpha, beta, x_tr) - y_tr)**2))

        mape_out.append(np.mean(np.absolute((y_val - yhat(alpha, beta, x_val))/y_val))*100)
        mape_in.append(np.mean(np.absolute((y_tr - yhat(alpha, beta, x_tr))/y_tr))*100)

        mae_out.append(np.mean(np.absolute(y_val - yhat(alpha, beta, x_val))))
        mae_in.append(np.mean(np.absolute(y_tr - yhat(alpha, beta, x_tr))))

        me_out.append(np.mean(y_val - yhat(alpha, beta, x_val)))
        me_in.append(np.mean(y_tr - yhat(alpha, beta, x_tr)))

    error = np.mean([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)
    std = np.std([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)

    return error, std

# SVR RBF
def svr_mse(x, y, i_mix):

    mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in = [], [], [], [], [], [], [], []
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
        svr = GridSearchCV(SVR(kernel='rbf'),para_grid)
        svr.fit(x_tr, y_tr)

        mse_out.append(np.mean((svr.predict(x_val) - y_val)**2))
        mse_in.append(np.mean((svr.predict(x_tr) - y_tr)**2))

        mape_out.append(np.mean(np.absolute((y_val - svr.predict(x_val))/y_val))*100)
        mape_in.append(np.mean(np.absolute((y_tr - svr.predict(x_tr))/y_tr))*100)

        mae_out.append(np.mean(np.absolute(y_val - svr.predict(x_val))))
        mae_in.append(np.mean(np.absolute(y_tr - svr.predict(x_tr))))

        me_out.append(np.mean(y_val - svr.predict(x_val)))
        me_in.append(np.mean(y_tr - svr.predict(x_tr)))

    error = np.mean([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)
    std = np.std([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)

    return error, std

# SVR poly
def svrp_mse(x, y, i_mix):

    mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in = [], [], [], [], [], [], [], []
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
        svr = GridSearchCV(SVR(kernel='poly'),para_grid)
        svr.fit(x_tr, y_tr)

        mse_out.append(np.mean((svr.predict(x_val) - y_val)**2))
        mse_in.append(np.mean((svr.predict(x_tr) - y_tr)**2))

        mape_out.append(np.mean(np.absolute((y_val - svr.predict(x_val))/y_val))*100)
        mape_in.append(np.mean(np.absolute((y_tr - svr.predict(x_tr))/y_tr))*100)

        mae_out.append(np.mean(np.absolute(y_val - svr.predict(x_val))))
        mae_in.append(np.mean(np.absolute(y_tr - svr.predict(x_tr))))

        me_out.append(np.mean(y_val - svr.predict(x_val)))
        me_in.append(np.mean(y_tr - svr.predict(x_tr)))

    error = np.mean([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)
    std = np.std([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)

    return error, std

# SFA
def sfa_mse(x, y, i_mix):

    mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in = [], [], [], [], [], [], [], []
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

        sfa = linear_model.LinearRegression()
        sfa.fit(np.log(x_tr), np.log(y_tr))

        y_hat_val = np.exp(sfa.predict(np.log(x_val)))
        y_hat_tr = np.exp(sfa.predict(np.log(x_tr)))

        mse_out.append(np.mean((y_hat_val - y_val)**2))
        mse_in.append(np.mean((y_hat_tr - y_tr)**2))

        mape_out.append(np.mean(np.absolute((y_val - y_hat_val)/y_val))*100)
        mape_in.append(np.mean(np.absolute((y_tr - y_hat_tr)/y_tr))*100)

        mae_out.append(np.mean(np.absolute(y_val - y_hat_val)))
        mae_in.append(np.mean(np.absolute(y_tr - y_hat_tr)))

        me_out.append(np.mean(y_val - y_hat_val))
        me_in.append(np.mean(y_tr - y_hat_tr))

    error = np.mean([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)
    std = np.std([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)

    return error, std

# CNLS
def cnls_mse(x, y, i_mix):

    mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in = [], [], [], [], [], [], [], []
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

        mse_out.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))
        mse_in.append(np.mean((yhat(alpha, beta, x_tr) - y_tr)**2))

        mape_out.append(np.mean(np.absolute((y_val - yhat(alpha, beta, x_val))/y_val))*100)
        mape_in.append(np.mean(np.absolute((y_tr - yhat(alpha, beta, x_tr))/y_tr))*100)

        mae_out.append(np.mean(np.absolute(y_val - yhat(alpha, beta, x_val))))
        mae_in.append(np.mean(np.absolute(y_tr - yhat(alpha, beta, x_tr))))

        me_out.append(np.mean(y_val - yhat(alpha, beta, x_val)))
        me_in.append(np.mean(y_tr - yhat(alpha, beta, x_tr)))

    error = np.mean([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)
    std = np.std([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)

    return error, std


# LCR
def lcr(x, y, i_mix):

    l = [0.1, 1, 2, 5, 8, 15]
    l_one = toolbox.L_opt(x, y, kfold, l)[1]

    mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in = [], [], [], [], [], [], [], []
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

        alpha, beta, epsilon = LCR.LCR(y_tr, x_tr, l_one)

        mse_out.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))
        mse_in.append(np.mean((yhat(alpha, beta, x_tr) - y_tr)**2))

        mape_out.append(np.mean(np.absolute((y_val - yhat(alpha, beta, x_val))/y_val))*100)
        mape_in.append(np.mean(np.absolute((y_tr - yhat(alpha, beta, x_tr))/y_tr))*100)

        mae_out.append(np.mean(np.absolute(y_val - yhat(alpha, beta, x_val))))
        mae_in.append(np.mean(np.absolute(y_tr - yhat(alpha, beta, x_tr))))

        me_out.append(np.mean(y_val - yhat(alpha, beta, x_val)))
        me_in.append(np.mean(y_tr - yhat(alpha, beta, x_tr)))

    error = np.mean([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)
    std = np.std([mse_out, mse_in, mape_out, mape_in, mae_out, mae_in, me_out, me_in], axis=1)

    return error, std


# load data
data = pd.read_csv('California.csv')
df = data.loc[data['Area'] == 'NEAR BAY']
df = df.sample(frac=0.2, random_state=42)
features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
x = df.loc[:, features]
y = df.loc[:, 'MEDV']
x = np.array(x)
y = np.array(y)

# validation setting
kfold = 5
random.seed(0)
i_mix = random.sample(range(len(y)), k=len(y))

data = lcr(x, y, i_mix)
df = pd.DataFrame(data)
df.to_excel('measure_lcr.xlsx')