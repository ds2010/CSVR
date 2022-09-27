import sys
sys.path.append('../functionall/')
import numpy as np
import pandas as pd
import random
import MC


def main():

    #simulations
    re_all = []
    for _ in range(M):
        re_all.append(MC.simulation_out(n, d, sig, nt))

    data = np.array(re_all)
    df = pd.DataFrame(data, columns = ['csvr', 'svr', 'cnls', 'lcr'])
    df.to_csv('mse_' + '{0}_{1}_{2}.csv'.format(n, d, sig), index=False)


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    M=50
    n=500
    d=5
    sig = 0.5
    nt = 1000

    main()