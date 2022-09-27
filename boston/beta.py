import sys
sys.path.append('../functionall/')
import numpy as np
import pandas as pd
from constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
import CSVR, CNLS, LCR
import toolbox


def beta_val(x, y):
    
    # CSVR
    alpha, beta, ksia, ksib = CSVR.CSVR(y, x, epsilon=3.6, u=3.7)
    csvr_cof = np.concatenate((beta, alpha.reshape(len(alpha), 1)), axis=1)
    csvr_val = np.array([np.mean(csvr_cof, axis=0), np.std(csvr_cof, axis=0), np.min(csvr_cof, axis=0), np.max(csvr_cof, axis=0)])

    # LCR
    alpha, beta, epsilon = LCR.LCR(y, x, L=0.45)
    lcr_cof = np.concatenate((beta, alpha.reshape(len(alpha), 1)), axis=1)
    lcr_val = np.array([np.mean(lcr_cof, axis=0), np.std(lcr_cof, axis=0), np.min(lcr_cof, axis=0), np.max(lcr_cof, axis=0)])

    #CNLS
    model = CNLS.CNLS(y, x, z=None, cet= CET_ADDI, fun= FUN_PROD, rts= RTS_VRS)
    model.optimize(OPT_LOCAL)
    alpha, beta = model.get_alpha(), model.get_beta()
    cnls_cof = np.concatenate((beta, alpha.reshape(len(alpha), 1)), axis=1)
    cnls_val = np.array([np.mean(cnls_cof, axis=0), np.std(cnls_cof, axis=0), np.min(cnls_cof, axis=0), np.max(cnls_cof, axis=0)])

    val = np.concatenate((csvr_val.T, lcr_val.T, cnls_val.T), axis=1)
    
    return val


data = pd.read_csv('Boston.csv')
x = data.loc[:, ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = data['MEDV']

x = np.array(x)
y = np.array(y)
cof = beta_val(x,y)
df = pd.DataFrame(cof, index=['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'alpha'], 
                    columns=['csvr_mean', 'csvr_std', 'csvr_min', 'csvr_max', 
                                'lcr_mean', 'lcr_std', 'lcr_min', 'lcr_max',
                                'cr_mean', 'cr_std', 'cr_min', 'cr_max'])
df.to_excel('beta.xlsx')