import numpy as np
import random
import CSVR, LCR


# Calculate yhat in testing sample
def yhat(alpha, beta, x_test):

    # compute yhat for each testing observation
    yhat = np.zeros((len(x_test),))
    for i in range(len(x_test)):
        yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).min(axis=0)
        
    return yhat


# calculate the index of training set
def index_tr(k, i_kfold):
    
    i_kfold_without_k = i_kfold[:k] + i_kfold[(k + 1):]
    flatlist = [item for elem in i_kfold_without_k for item in elem]

    return flatlist


# cross validation: find the optimal u using: 
#  1) usual rule; 2) one standard error rule
def u_opt(x, y, kfold, epsilon, u):

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
        for j in u:
            alpha, beta, ksia, ksib = CSVR.CSVR(y=y_tr, x=x_tr, epsilon=epsilon, u=j)
            error_tmp.append( (yhat(alpha, beta, x_val) - y_val)**2 )

        error.append(np.array(error_tmp).reshape(len(u), len(y_val)).T)
        
    error = np.array(error).reshape(len(y), len(u))

    # compute average error over all folds 
    cv = np.mean(error, axis=0)
    
    # one standard error rule
    errs0 = np.zeros((kfold, len(u)))
    for k in range(kfold):
            i_val = i_kfold[k]
            errs0[k, :] = np.mean(error[i_val, :], axis=0)

    # standard errors
    se = np.std(errs0, ddof=1, axis=0) / np.sqrt(kfold)  

    # largest value of u such that error is within one standard error of the cross-validated errors for u.min.
    i2 = np.argmax(np.where(cv <= cv[np.argmin(cv)]+se[np.argmin(cv)]))
        
    # usual rule
    i1 = np.argmin(cv)

    return u[i1], u[i2]


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

    return L_para[i2]

# cross validation using grid search
def GridSearch(x, y, kfold, epsilon, u):
    # resample the index 
    i_mix = random.sample(range(len(y)), k=len(y))

    # divide up i.mix into K equal size chunks
    m = len(y) // kfold
    i_kfold = [i_mix[i:i+m] for i in range(0, len(i_mix), m)]

    # total errors in each fold 
    error_graph = []
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

        graph_x = []
        graph_y = []
        error_tmp = []
        for i in epsilon:
            graph_x_row = []
            graph_y_row = []
            error_tmp_row = []
            for j in u:
                alpha, beta, ksia, ksib = CSVR.CSVR(y=y_tr, x=x_tr, epsilon=i, u=j)
                error_tmp_row.append( np.mean((yhat(alpha, beta, x_val) - y_val)**2 ))
                graph_x_row.append(i)
                graph_y_row.append(j)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            error_tmp.append(error_tmp_row)

        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        error_graph.append(error_tmp)
        
    mse_graph = np.mean(np.array(error_graph), axis=0)

    # search the position of parameters
    pos_min_mse = np.argwhere(mse_graph == np.min(mse_graph))[0]

    # return epsilon and u
    return graph_x[pos_min_mse[0],pos_min_mse[1]], graph_y[pos_min_mse[0],pos_min_mse[1]]
