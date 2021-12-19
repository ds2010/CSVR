import numpy as np


def inputs(n, d):

	sig = 0.4**0.5

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
