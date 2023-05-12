import sys
sys.path.append('../functionall/')
import numpy as np
import pandas as pd
import random
import os
import MC
from multiprocessing import Pool


if 'SLURM_CPUS_PER_TASK' in os.environ:
    cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    print("Dectected %s CPUs through slurm"%cpus)
else:
    # None means that it will auto-detect based on os.cpu_count()
    cpus = None
    print("Running on default number of CPUs (default: all=%s)"%os.cpu_count())

def process(arg):

    mse = MC.simulation_out(n, d, sig, nt)
    return mse

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    n=500
    d=2
    sig = 2
    nt = 1000

    with Pool(cpus) as p:
        
        df = pd.DataFrame(p.map(process, range(50)), columns = ['mse_csvr', 'mse_svr', 'mse_svrl', 'mse_cnls', 'mse_lcr', 
                                        'mape_csvr', 'mape_svr', 'mape_svrl', 'mape_cnls', 'mape_lcr'])
        df.to_csv('measure' + '{0}_{1}_{2}.csv'.format(n, d, sig))