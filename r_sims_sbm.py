import nd_python_avon as nd_p 
import numpy as np
import glob
import os
import re
import json
import sklearn.mixture
import math

n = 100_000
num_networks= 20

# taus1 = np.arange(.00015, .0055, 0.0001)
# taus2 = np.arange(.0055, .0725, .001)
# taus3 = np.arange(.0725, .5, .005)
# taus = np.concatenate((taus1, taus2, taus3))

## taus placing 24 points evenly across R0 = 0.8-5.5, by inverting the
## measured tau -> R0 curve of the runs already on disk; see the tau
## coverage section of thesis_figs.ipynb
taus = np.array([0.022212, 0.026052, 0.036603, 0.043788, 0.047913, 0.058698, 0.062568, 0.080724, 0.085586, 0.097463, 0.111, 0.11973, 0.12341, 0.12708, 0.16091, 0.17063, 0.19025, 0.19558, 0.22943, 0.24267, 0.26836, 0.28281, 0.32692, 0.34318])

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['reconnect']


SUFFIX = '_sbm'   # duration+ages/seir_sims/{data}_{k}{SUFFIX}_fin.json


def claim_index(data):
    """Reserve the next unused replicate index, and create the file to hold it.

    New runs continue past whatever is already on disk instead of overwriting it.
    The tau grid above is not the one earlier batches used, and the aggregation in
    paper_figs.ipynb / thesis_figs.ipynb keys on the tau value rather than on its
    position in the grid, so old and new files pool together without conflict.

    The empty placeholder claims the index immediately, so that a resubmitted job --
    or a second job on the same pair -- cannot pick the same one while this network
    is still simulating.  Aggregators skip files they cannot parse, so a placeholder
    left behind by a job that died is ignored rather than counted.
    """
    pattern = re.compile(rf'^{data}_(\d+){SUFFIX}_fin\.json$')
    used = [-1]
    for path in glob.glob(f'duration+ages/seir_sims/{data}_*_fin.json'):
        match = pattern.match(os.path.basename(path))
        if match:
            used.append(int(match.group(1)))
    k = max(used) + 1
    open(f'duration+ages/seir_sims/{data}_{k}{SUFFIX}_fin.json', 'w').close()
    return k


for i, data in enumerate(datas):
    with open(f'input_data/gmm/optimal_components_{data}_log.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}.json', 'r') as f:
        egos = json.load(f)
    cm = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')

    for _ in range(num_networks):
        k = claim_index(data)
        print(f'network {k} for data {data}', flush=True)
        res = nd_p.sbm_gillesp(contact_matrix=cm, partitions=partitions, taus=taus, iterations=48, num_infec=1)
        with open(f'duration+ages/seir_sims/{data}_{k}{SUFFIX}_fin.json','w') as f:
            json.dump(res, f)
