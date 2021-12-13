import numpy as np
import CSVR
import LCR
import pystoned.CNLS as CNLS
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from sklearn.svm import SVR


def inputs(n, d, sig):

	x = np.random.uniform(low=1, high=10, size=(n, d))
	nse = np.random.normal(0, sig, n)

	if d == 1:
		y_true = 3 + x[:,0]**0.5
		y = y_true + nse
	elif d == 2:
		y_true = 3 + x[:,0]**0.2 + x[:,1]**0.3
		y = y_true + nse
	else:
		y_true = 3 + x[:, 0]**0.05 + x[:, 1]**0.15 + x[:, 2]**0.3
		y = y_true + nse

	return x, y, y_true


def main(sim):

	n = 100
	d = 3
	sig = 1

	mse_csvr, mse_svr, mse_cnls, mse_lcr = [], [], [], []

	for i in range(sim):

		# DGP
		x, y, y_true = inputs(n, d, sig)

		# solve the CSVR model
		alpha, beta, ksia, ksib = CSVR.CSVR(y, x, epsilon=0.4, u=1.5) #6.9, 8.3
		y_csvr = alpha + np.sum(beta * x, axis=1)
		mse_csvr.append(np.mean((y_true - y_csvr)**2))

		# solve the SVR model
		svr = SVR(C=1.5, epsilon=0.4).fit(x, y)
		y_svr = svr.predict(x)
		mse_svr.append(np.mean((y_true - y_svr)**2))

		# solve the CNLS model
		model = CNLS.CNLS(y, x, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
		model.optimize(OPT_LOCAL)
		mse_cnls.append(np.mean((model.get_frontier() - y_true)**2))

		# solve the LCR model
		a, b, epsilon = LCR.LCR(y, x, L=1.3)
		y_lcr = a + np.sum(b * x, axis=1)
		mse_lcr.append(np.mean((y_true - y_lcr)**2))

		mse = [np.mean(mse_csvr), np.mean(mse_cnls), np.mean(mse_lcr), np.mean(mse_svr)]

	return mse


if __name__ == '__main__':

    np.random.seed(0)
    sim = 50

    with open('01' + '.txt', 'w') as f:
        for item in main(sim):
            f.write("%s\n" % item)