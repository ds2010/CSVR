import numpy as np
import random

# modules for cross validation
import CSVR
import LCR

# Calculate yhat in testing sample
def yhat(alpha, beta, x_test):

    # compute yhat for each testing observation
    yhat = np.zeros((len(x_test), 1))
    for i in range(len(x_test)):
        yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).min(axis=0)
        
    return yhat


# calculate the index of training set
def index_tr(k, i_kfold):
    if k == 0:
        return i_kfold[1]+i_kfold[2]+i_kfold[3]+i_kfold[4]
    elif k == 1:
        return i_kfold[0]+i_kfold[2]+i_kfold[3]+i_kfold[4]
    elif k == 2:
        return i_kfold[0]+i_kfold[1]+i_kfold[3]+i_kfold[4]
    elif k == 3:
        return i_kfold[0]+i_kfold[1]+i_kfold[2]+i_kfold[4]
    else:
        return i_kfold[0]+i_kfold[1]+i_kfold[2]+i_kfold[3]


# cross validation: find the optimal u using: 
#  1) usual rule; 2) one standard error rule
def u_opt(x, y, kfold, u_para):

    # resample the index 
    i_mix = random.sample(range(len(y)), k=len(y))

    # divide up i.mix into K equal size chunks
    m = len(y) // kfold
    i_kfold = [i_mix[i:i+m] for i in range(0, len(i_mix), m)]

    # total errors in each fold 
    error = []
    for k in range(kfold):
        print("Fold", k, "\n")

        i_tr = index_tr(k, i_kfold)
        i_val = i_kfold[k]

        # training predictors, training responses
        x_tr = x[i_tr, :]  
        y_tr = y[i_tr]   
        # validation predictors, validation responses
        x_val = x[i_val, :]
        y_val = y[i_val]  

        error_tmp = []
        for j in u_para:
            alpha, beta, ksia, ksib = CSVR.CSVR(y=y_tr, x=x_tr, epsilon=0.5, u=j)
            error_tmp.append( (yhat(alpha, beta, x_val) - y_val)**2 )

        error.append(np.array(error_tmp).reshape(len(u_para), len(y_val)).T)
        
    error = np.array(error).reshape(len(y), len(u_para))

    # compute average error over all folds 
    cv = np.mean(error, axis=0)
    
    # one standard error rule
    errs0 = np.zeros((kfold, len(u_para)))
    for k in range(kfold):
            i_val = i_kfold[k]
            errs0[k, :] = np.mean(error[i_val, :], axis=0)

    # standard errors
    se = np.std(errs0, ddof=1, axis=0) / np.sqrt(kfold)  

    # largest value of u such that error is within one standard error of the cross-validated errors for u.min.
    i2 = np.argmax(np.where(cv <= cv[np.argmin(cv)]+se[np.argmin(cv)]))
        
    # usual rule
    i1 = np.argmin(cv)
    return u_para[i1], u_para[i2]


# cross validation: find the optimal epcilon
def e_opt(x, y, kfold, e_para):

    # resample the index 
    i_mix = random.sample(range(len(y)), k=len(y))

    # divide up i.mix into K equal size chunks
    m = len(y) // kfold
    i_kfold = [i_mix[i:i+m] for i in range(0, len(i_mix), m)]

    # total errors in each fold 
    error = []
    for k in range(kfold):
        print("Fold", k, "\n")

        i_tr = index_tr(k, i_kfold)
        i_val = i_kfold[k]

        # training predictors, training responses
        x_tr = x[i_tr, :]  
        y_tr = y[i_tr]   
        # validation predictors, validation responses
        x_val = x[i_val, :]
        y_val = y[i_val]  

        error_tmp = []
        for j in e_para:
            alpha, beta, ksia, ksib = CSVR.CSVR(y=y_tr, x=x_tr, epsilon=j, u=4)
            error_tmp.append( (yhat(alpha, beta, x_val) - y_val)**2 )

        error.append(np.array(error_tmp).reshape(len(e_para), len(y_val)).T)
        
    error = np.array(error).reshape(len(y), len(e_para))

    # compute average error over all folds 
    cv = np.mean(error, axis=0)
    
    # one standard error rule
    errs0 = np.zeros((kfold, len(e_para)))
    for k in range(kfold):
            i_val = i_kfold[k]
            errs0[k, :] = np.mean(error[i_val, :], axis=0)

    # standard errors
    se = np.std(errs0, ddof=1, axis=0) / np.sqrt(kfold)  

    # largest value of u such that error is within one standard error of the cross-validated errors for epsilon.min.
    i2 = np.argmax(np.where(cv <= cv[np.argmin(cv)]+se[np.argmin(cv)]))
        
    # usual rule
    i1 = np.argmin(cv)
    return e_para[i1], e_para[i2]

# cross validation for LCR: find the optimal L using: 
#  1) usual rule; 2) one standard error rule
def L_opt(x, y, kfold, L_para):

    # resample the index 
    i_mix = random.sample(range(len(y)), k=len(y))

    # divide up i.mix into K equal size chunks
    m = len(y) // kfold
    i_kfold = [i_mix[i:i+m] for i in range(0, len(i_mix), m)]

    # total errors in each fold 
    error = []
    for k in range(kfold):
        print("Fold", k, "\n")

        i_tr = index_tr(k, i_kfold)
        i_val = i_kfold[k]

        # training predictors, training responses
        x_tr = x[i_tr, :]  
        y_tr = y[i_tr]   
        # validation predictors, validation responses
        x_val = x[i_val, :]
        y_val = y[i_val]  

        error_tmp = []
        for j in L_para:
            alpha, beta, epsilon = LCR.LCR(y=y_tr, x=x_tr, L=j)
            error_tmp.append( (yhat(alpha, beta, x_val) - y_val)**2 )

        error.append(np.array(error_tmp).reshape(len(L_para), len(y_val)).T)
        
    error = np.array(error).reshape(len(y), len(L_para))

    # compute average error over all folds 
    cv = np.mean(error, axis=0)
    
    # one standard error rule
    errs0 = np.zeros((kfold, len(L_para)))
    for k in range(kfold):
            i_val = i_kfold[k]
            errs0[k, :] = np.mean(error[i_val, :], axis=0)

    # standard errors
    se = np.std(errs0, ddof=1, axis=0) / np.sqrt(kfold)  

    # largest value of u such that error is within one standard error of the cross-validated errors for u.min.
    i2 = np.argmax(np.where(cv <= cv[np.argmin(cv)]+se[np.argmin(cv)]))
        
    # usual rule
    i1 = np.argmin(cv)
    return L_para[i1], L_para[i2]