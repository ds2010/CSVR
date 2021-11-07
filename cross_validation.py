import numpy as np
from sklearn.model_selection import KFold
import CSVR


# Calculate yhat in testing sample
def yhat(alpha, beta, X_test):

    # compute yhat for each testing observation
    yhat = np.zeros((len(X_test), 1))
    for i in range(len(X_test)):
        yhat[i] = (alpha + np.matmul(beta, X_test[i].T)).min(axis=0)
        
    return yhat


# Cross validation estimate
def cross_validation(X, y, n_fold, u_parameter):

    # standardize variables x and y
    #X = (X - np.mean(X, axis=0)) / np.linalg.norm(X - np.mean(X, axis=0), 2)
    #y = y/np.linalg.norm(y, 2)
    data = np.concatenate((X, y), axis=1)

    # resample the data
    kfold = KFold(n_splits=n_fold, random_state=1, shuffle=True)

    # calculate MSE in each u parameter
    MSE = []
    for j in u_parameter:
        # calculate MSE_0 in each fold
        MSE_0 = []
        for train_index, test_index in kfold.split(data):
            # split the data 
            data_train, data_test = data[train_index], data[test_index]

            # estimate the CSVR model
            alpha_tmp, beta_tmp, ksia, ksib = CSVR.CSVR(y=data_train[:, -1], x=data_train[:, :-1], epsilon=10^-3, u=j)
            
            # calculate MSE
            mse_tmp = np.sum( (data_test[:, -1].reshape(len(data_test), 1) - yhat(alpha_tmp, beta_tmp, data_test[:, :-1]) )**2, axis=0)/len(data_test)
            MSE_0.append(mse_tmp)

        MSE.append(MSE_0)

    # return array with error measures per value of u (rows) and fold (columns)
    return np.array(MSE).reshape(len(u_parameter), n_fold)


# find the optimal u using one standard error rule
def stand_error(X, y, n_fold, u_parameter):

    # cross validation 
    test = cross_validation(X, y, n_fold, u_parameter)

    # Index of u_parameter with minimum average error measure
    ind0 = np.argmin(np.mean(test, axis=1))

    # Minimum average error measure
    cv_mean_opt0 = np.mean(test[ind0, :])  

    # Standard error of error measures for value of lambda that minimizes average error measure
    cv_se_opt0 = np.std(test[ind0, :], ddof=1) / np.sqrt(n_fold) 

    # Maximum value of lambda such that average error measure is still
    # smaller than minimum average error measure plus 1 standard error
    one_se_rule_i = np.argmax(u_parameter * (np.mean(test, axis=1) <= cv_mean_opt0 + cv_se_opt0))
    u_opt = u_parameter[one_se_rule_i]

    return u_opt
