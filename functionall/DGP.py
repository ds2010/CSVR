import numpy as np


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

def Gauss(n, d, SNR):

	x = np.random.uniform(low=-1, high=1, size=(n, d))
	y_true = np.linalg.norm(x, axis=1)**2

	sigma = np.sqrt(np.var(y_true, ddof=1, axis=0)/SNR)
	nse = np.random.normal(0, sigma, n)

	y = y_true + nse

	# normalization
	normalization = np.sqrt(np.sum(x**2, axis=0))/np.sqrt(x.shape[0])
	x = x/normalization

	return x, y, y_true