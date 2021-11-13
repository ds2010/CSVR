import pandas as pd
import numpy as np
import random
import toolbox


# load data
data = pd.read_csv('Boston.csv')
X = np.array(data.loc[:, ['NOX', 'DIS', 'RM', 'PTRATIO']])[:500, :]
y = np.array(data['MEDV'])
y = y.reshape(len(y), 1)[:500, :]
X = np.log(X)
y = np.log(y)

# parameter
u_para = np.linspace(0.01, 10, 50)
kfold = 5

random.seed(0)
# usual rule
#u_opt = toolbox.u_opt(X, y, kfold, u_para, method=False)
#print(u_opt)
# (38, 7.75734693877551)

# one standard error rule
u_opt = toolbox.u_opt(X, y, kfold, u_para, method=True)
print(u_opt)
