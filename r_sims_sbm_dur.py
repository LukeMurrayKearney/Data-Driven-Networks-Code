# import sys
# import os
# sys.path.append(os.path.abspath('..'))
import nd_python_avon as nd_p 
import numpy as np
import glob
import os
import re
import json
import sklearn.mixture
import math

n = 100_000
num_networks= 40

# taus1 = np.arange(0.01,0.1,0.01)
# taus2 = np.arange(0.1,0.2,0.02)
# taus3 = np.arange(0.2,0.6,0.05)
# taus = np.concatenate((taus1, taus2, taus3))

## taus placing 24 points evenly across R0 = 0.8-5.5, by inverting the
## measured tau -> R0 curve of the runs already on disk; see the tau
## coverage section of thesis_figs.ipynb
taus = np.array([0.16296, 0.24415, 0.3156, 0.41119, 0.53399, 0.67438, 0.86986, 1.1012, 1.4404, 1.7394, 2.2201, 2.5515, 3.3123, 4.0838, 5.0516, 6.2486, 7.7292, 9.5608, 11.826, 14.629, 18.095, 22.383, 27.687, 34.247])

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['reconnect']

model = 'sbm_dur'


def make_contact_matrices(egos, num_durs):
    num_per_bucket = np.zeros(np.max([a['age'] for a in egos])+1)
    contact_matrix = [np.zeros((np.max([a['age'] for a in egos])+1, np.max([a['age'] for a in egos])+1)) for _ in range(num_durs)]
    for ego in egos:
        num_per_bucket[ego['age']] += 1
        for j, val in enumerate(ego['contacts']):
            contact_matrix[j%num_durs][ego['age'], j//num_durs] += val
    for j in range(num_durs):
        contact_matrix[j] = np.divide(contact_matrix[j].T, num_per_bucket).T
        contact_matrix[j] = (contact_matrix[j] + contact_matrix[j].T)/2
    return contact_matrix, num_per_bucket

SUFFIX = '_sbm_dur'   # duration+ages/seir_sims/{data}_{k}{SUFFIX}_fin.json


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
    with open(f'duration+ages/data/gmm_opt_comp/optimal_components_{data}_log_smalldur.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}_dur_small.json', 'r') as f:
        egos = json.load(f)
    props = np.genfromtxt(f'input_data/durations/{data}.csv', delimiter=',')

    contact_matrix, num_per_bucket = make_contact_matrices(egos, num_durs=3)
    
    for _ in range(num_networks):
        k = claim_index(data)
        print(f'network {k} for data {data}', flush=True)
        res = nd_p.sbm_gillesp_dur(contact_matrix=contact_matrix, num_dur=3, partitions=partitions, taus=taus, iterations=48, props=props.tolist(), num_infec=1)
        with open(f'duration+ages/seir_sims/{data}_{k}{SUFFIX}_fin.json','w') as f:
            json.dump(res, f)
