import sys
from xmlrpc.server import ServerHTMLDoc

from sklearn import svm
sys.path.append('../functionall/')

import numpy as np
import random
import pandas as pd
import CNLS, CSVR, toolbox
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS, FUN_COST
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,SGDRegressor
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV,cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
import matplotlib.pyplot as plt


np.random.seed(0)
random.seed(0)

# Calculate R2
def R2(y_hat, y):
    y_bar = np.mean(y)
    RSS = np.sum((y-y_hat)**2)
    TSS = np.sum((y-y_bar)**2)

    return 1 - RSS/TSS


# Calculate yhat in testing sample
def yhat(alpha, beta, x_test):

    # compute yhat for each testing observation
	yhat = np.zeros((len(x_test),))
	for i in range(len(x_test)):
		yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).max(axis=0)
        
	return yhat


# load data
data = pd.read_csv('electricityFirms.csv')
x = data.loc[:,["Energy","Length","Customers"]]
y = data["TOTEX"]

# data = pd.read_csv('cpus.txt', names=['name','Model','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP'])
# x = data.loc[:,['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX']]
# y = data['PRP']

# data = pd.read_csv('winequality-red.csv')
# x = data.drop('quality', axis = 1)
# y = data['quality']

x = np.array(x)
y = np.array(y)

# kf = KFold(n_splits=5)
# l_regression = SVR()
# scores = cross_val_score(l_regression, x, y, cv=kf, scoring='r2')
# print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))



# model = LinearRegression()
# lr = model.fit(x_train,y_train)
# y_train_pred = lr.predict(x_train)
# y_test_pred = lr.predict(x_test)

# print(lr.score(x_test,y_test))


# alpha, beta, ksia, ksib = CSVR.CSVR(y_tr, x_tr, epsilon=0.1, u=2)

# x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
# model = CNLS.CNLS(y, x, z=None, cet= CET_ADDI, fun= FUN_COST, rts= RTS_VRS)
# model.optimize(OPT_LOCAL)
# alpha, beta = model.get_alpha(), model.get_beta()
alpha, beta, ksia, ksib = CSVR.CSVR(y, x, epsilon=0.1, u=2)

mse = np.mean((yhat(alpha, beta, x) - y)**2)
# r2 = R2(yhat(alpha, beta, x_test), y_test)
print(mse) #CSVR: 0.4543