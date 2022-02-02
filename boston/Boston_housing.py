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

	# u = np.array([0.1, 0.5, 1, 2, 5])
	# epsilon = np.array([0, 0.001, 0.01, 0.1, 0.2])
	# e_grid, u_grid = toolbox.GridSearch(x, y, kfold, epsilon=epsilon, u=u)
	
	
	error_all = []
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

		alpha, beta, ksia, ksib = CSVR.CSVR(y_tr, x_tr, epsilon=0.1, u=3)

		error_all.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))
		
	mse = np.mean(np.array(error_all), axis=0)
	# std = np.std(np.array(error_all), axis=0)
	# mdi = np.median(np.array(error_all), axis=0)

	return mse

# SVR
def svr_mse(x, y, i_mix):

	error_all = []
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

		# para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'epsilon': [0, 0.001, 0.01, 0.1, 0.2]}
		svr = SVR(kernel='rbf', C=1, gamma=0.1)
		svr.fit(x_tr, y_tr)

		error_all.append(np.mean((svr.predict(x_val) - y_val)**2))

	mse = np.mean(np.array(error_all), axis=0)

	return mse

# CNLS
def cnls_mse(x, y, i_mix):

	error_all = []
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

		error_all.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))

	mse = np.mean(np.array(error_all), axis=0)

	return mse

# lasso
def lasso(x, y, i_mix):

	error_all = []
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

		clf = linear_model.Lasso()
		clf.fit(x_tr, y_tr)

		error_all.append(np.mean((clf.predict(x_val) - y_val)**2))

	mse = np.mean(np.array(error_all), axis=0)

	return mse

# convex regression
def lcr(x, y, i_mix):

	error_all = []
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

		alpha, beta, epsilon = LCR.LCR(y_tr, x_tr, L=1)

		error_all.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))

	mse = np.mean(np.array(error_all), axis=0)

	return mse


# load data
data = pd.read_csv('Boston.csv')
# data = data[~(data['MEDV'] >= 50.0)]

x = data.loc[:, ['NOX', 'DIS', 'PTRATIO', 'INDUS', 'TAX', 'ZN', 'RAD']]
# x = data.loc[:, ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = data['MEDV']

# y = np.log1p(y)
# for col in ['NOX', 'DIS', 'PTRATIO']:
# 	x[col] = np.log1p(x[col])
# for col in x.columns:
#     if np.abs(x[col].skew()) > 0.3:
#         x[col] = np.log1p(x[col])

x = np.array(x)
y = np.array(y)


kfold = 5
i_mix = random.sample(range(len(y)), k=len(y))
# print(i_mix)

mse_csvr = csvr_mse(x, y, i_mix)
mse_cnls = cnls_mse(x, y, i_mix)
mse_svr = svr_mse(x, y, i_mix)
# mse_lasso = lasso(x, y, i_mix)
mse_lcr = lcr(x, y, i_mix)
print(mse_csvr, mse_svr, mse_cnls, mse_lcr)