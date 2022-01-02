import numpy as np
import toolbox, DGP
import random
import matplotlib.pyplot as plt
import time

start_time = time.time()


np.random.seed(0)
random.seed(0)

n = 100
d = 3
sig = 0.4

x, y, y_true = DGP.inputs(n, d, sig)


kfold = 10
u_para = np.linspace(0.01, 10, 20)
epsilon = np.array([0, 0.001, 0.01, 0.1, 0.2])

a, b= toolbox.GridSearch(x, y, kfold, epsilon=epsilon, u_para=u_para)
print(a,b)
print('----- %s second -----' % (time.time() - start_time))
# plt.plot(u_para, mse)
# plt.savefig('mse.png')
# plt.show()
