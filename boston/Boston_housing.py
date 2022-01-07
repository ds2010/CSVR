import sys
sys.path.append('../functionall/')
import pandas as pd
import numpy as np
import random
import CNLS, CSVR, toolbox
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


# Calculate yhat in testing sample
def yhat(alpha, beta, x_test):

    # compute yhat for each testing observation
	yhat = np.zeros((len(x_test),))
	for i in range(len(x_test)):
		yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).min(axis=0)
        
	return yhat

# CSVR
def csvr_mse(x, y, i_kfold):

	u = np.array([0.1, 0.5, 1, 2, 5])
	epsilon = np.array([0, 0.001, 0.01, 0.1, 0.2])
	kfold = 10
	e_grid, u_grid = toolbox.GridSearch(x, y, kfold, epsilon=epsilon, u=u)
	error_all = []
	for k in range(kfold):
		print("Fold", k, "\n")

		i_tr = toolbox.index_tr(k, i_kfold)
		i_val = i_kfold[k]

		# training predictors, training responses
		x_tr = x[i_tr, :]  
		y_tr = y[i_tr]   

		# validation predictors, validation responses
		x_val = x[i_val, :]
		y_val = y[i_val] 

		alpha, beta, ksia, ksib = CSVR.CSVR(y_tr, x_tr, epsilon=e_grid, u=u_grid)

		error_all.append(np.mean((yhat(alpha, beta, x_val) - y_val)**2))
		
	mse = np.mean(np.array(error_all), axis=0)
	# std = np.std(np.array(error_all), axis=0)
	# mdi = np.median(np.array(error_all), axis=0)

	return mse

# SVR
def svr_mse(x, y, i_kfold):

	error_all = []
	kfold = 10
	for k in range(kfold):
		print("Fold", k, "\n")

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

		error_all.append(np.mean((svr.predict(x_val) - y_val)**2))

	mse = np.mean(np.array(error_all), axis=0)

	return mse

# CNLS
def cnls_mse(x, y, i_kfold):

	error_all = []
	kfold = 10
	for k in range(kfold):
		print("Fold", k, "\n")

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

np.random.seed(0)
random.seed(0)
# load data
data = pd.read_csv('Boston.csv')


# #Remove MEDV outliers (MEDV = 50.0)
# idx = data.loc[data['MEDV'] >= 50].index
# data = data.drop(idx)
data = data.head(20)

x = data.loc[:, ['NOX', 'DIS', 'RM', 'PTRATIO']]
y = data['MEDV']
for col in ['NOX', 'DIS']:
    x[col] = np.log1p(x[col])
x = np.array(x)
y = np.array(np.log1p(y))

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

kfold = 10


i_mix = random.sample(range(len(y)), k=len(y))

# divide up i.mix into K equal size chunks
m = len(y) // kfold
i_kfold = [i_mix[i:i+m] for i in range(0, len(i_mix), m)]


mse_csvr = csvr_mse(x, y, i_kfold)
mse_cnls = cnls_mse(x, y, i_kfold)
mse_svr = svr_mse(x, y, i_kfold)
print(mse_csvr, mse_svr, mse_cnls)
