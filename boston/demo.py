import sys
sys.path.append('../functionall/')

import numpy as np
import random
import pandas as pd
import CNLS, CSVR, toolbox
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


np.random.seed(0)
random.seed(0)

# Calculate yhat in testing sample
def yhat(alpha, beta, x_test):

    # compute yhat for each testing observation
	yhat = np.zeros((len(x_test),))
	for i in range(len(x_test)):
		yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).min(axis=0)
        
	return yhat


data = pd.read_csv('Boston.csv')

# #Remove MEDV outliers (MEDV = 50.0)
# idx = data.loc[data['MEDV'] >= 50].index
# data = data.drop(idx)
data = data.head(50)

x = data.loc[:, ['NOX', 'DIS', 'RM', 'PTRATIO']]
y = data['MEDV']
for col in ['NOX', 'DIS']:
    x[col] = np.log1p(x[col])
x = np.array(x)
y = np.array(np.log1p(y))

# training predictors, training responses
x_tr = x[0:40, :]  
y_tr = y[0:40]   

# validation predictors, validation responses
x_val = x[40:50, :]
y_val = y[40:50] 


# alpha, beta, ksia, ksib = CSVR.CSVR(y_tr, x_tr, epsilon=0.1, u=2)

model = CNLS.CNLS(y, x, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
model.optimize(OPT_LOCAL)
alpha, beta = model.get_alpha(), model.get_beta()

mse = np.mean((yhat(alpha, beta, x_val) - y_val)**2)
print(mse)