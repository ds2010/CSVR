import sys
sys.path.append('../functionall/')
import numpy as np
import pandas as pd
import random
import MC


def main():

    #simulations
    re_all = []
    for i in range(M):
        re_all.append(MC.simulation(n, d, sig))

    data = np.array(re_all)
    df = pd.DataFrame(data, columns = ['csvr', 'svr', 'cnls', 'lcr'])
    df.to_csv('mse' + '{0}_{1}_{2}.csv'.format(n, d, sig), index=False)


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    M=50
    n=50
    d=1
    sig = 1

    main()