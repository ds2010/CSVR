import numpy as np


def inputs(n, d, SNR):

	x = np.random.uniform(low=-1, high=1, size=(n, d))
	y_true = np.linalg.norm(x, axis=1)**2

	sigma = np.sqrt(np.var(y_true, ddof=1, axis=0)/SNR)
	nse = np.random.normal(0, sigma, n)
	
	y = y_true + nse

	# normalization
	normalization = np.sqrt(np.sum(x**2, axis=0))/np.sqrt(x.shape[0])
	x = x/normalization

	return x, y, y_true
