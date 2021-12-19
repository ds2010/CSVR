import numpy as np
import CNLS, CR, CSVR, LCR, DGP, toolbox
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def simulation(n, d, epsilon):
	
	kfold = 10
	para = np.linspace(0.1, 5, 50)

	# DGP
	x, y, y_true = DGP.inputs(n, d)

	# solve the CSVR model
	u_std, u_one = toolbox.u_opt(x, y.reshape(n, 1), kfold, epsilon, u_para=para)
	alpha, beta, ksia, ksib = CSVR.CSVR(y, x, epsilon=epsilon, u=u_one)
	y_csvr = alpha + np.sum(beta * x, axis=1)
	mse_csvr = np.mean((y_true - y_csvr)**2)

	# solve the SVR model
	para_grid = {'C': [0.1, 0.5, 1, 2, 5], 'kernel': ['rbf', 'linear']}
	svr = GridSearchCV(SVR(),para_grid)
	svr.fit(x, y)
	y_svr = svr.predict(x)
	mse_svr = np.mean((y_true - y_svr)**2)

	# solve the CNLS model
	model1 = CNLS.CNLS(y, x, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
	model1.optimize(OPT_LOCAL)
	mse_cnls = np.mean((model1.get_frontier() - y_true)**2)

	# solve the LCR model
	L_std, L_one = toolbox.L_opt(x, y.reshape(n, 1), kfold, L_para=para)
	a, b, epsilon = LCR.LCR(y, x, L=L_one)
	y_lcr = a + np.sum(b * x, axis=1)
	mse_lcr = np.mean((y_true - y_lcr)**2)

	# solve the CR model
	model2 = CR.CR(y, x, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
	model2.optimize(OPT_LOCAL)
	mse_cr = np.mean((model2.get_frontier() - y_true)**2)

	return mse_csvr, mse_svr, mse_cnls, mse_lcr, mse_cr
