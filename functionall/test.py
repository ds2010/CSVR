import numpy as np
import toolbox, DGP
import random
import matplotlib.pyplot as plt


np.random.seed(0)
random.seed(0)

n = 100
d = 3

x, y, y_true = DGP.inputs(n, d)


kfold = 10
u_para = np.linspace(0.001, 1, 50)

std, one, mse = toolbox.u_opt(x, y.reshape(n, 1), kfold, u_para)
plt.plot(u_para, mse)
plt.show()

