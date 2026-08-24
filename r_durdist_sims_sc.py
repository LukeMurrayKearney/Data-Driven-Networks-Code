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
num_networks= 50

# taus = np.array([0.059363310568763435, 0.10836004164411119, 0.1692509753028252])
# taus = np.linspace(0, .5, 10)
taus = np.array([0.12835, 0.17162, 0.2375, 0.30992, 0.35898, 0.50568, 0.5697, 0.72, 0.91214, 1.0013, 1.1238, 1.3562, 1.5168, 1.691, 1.9172, 2.3521, 2.3939, 2.7327, 3.4286, 3.6842, 3.7948, 4.2061, 4.5283, 4.7311])


buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['reconnect']


SUFFIX = ''   # duration+ages/seir_sims/{data}_{k}{SUFFIX}_age_dur.json


def claim_index(data):
    """Reserve the next unused replicate index, and create the file to hold it.

    New runs continue past whatever is already on disk instead of overwriting it.
    This loop used to start from 0 every time, so resubmitting the job destroyed the
    first `num_networks` results of the previous batch.

    The empty placeholder claims the index immediately, so that a resubmitted job --
    or a second job on the same pair -- cannot pick the same one while this network
    is still simulating.  Aggregators skip files they cannot parse, so a placeholder
    left behind by a job that died is ignored rather than counted.
    """
    pattern = re.compile(rf'^{data}_(\d+){SUFFIX}_age_dur\.json$')
    used = [-1]
    for path in glob.glob(f'duration+ages/seir_sims/{data}_*_age_dur.json'):
        match = pattern.match(os.path.basename(path))
        if match:
            used.append(int(match.group(1)))
    k = max(used) + 1
    open(f'duration+ages/seir_sims/{data}_{k}{SUFFIX}_age_dur.json', 'w').close()
    return k


for i, data in enumerate(datas):
    with open(f'duration+ages/data/gmm_opt_comp/optimal_components_{data}_log_smalldur.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}_dur_small.json', 'r') as f:
        egos = json.load(f)
    props = np.genfromtxt(f'input_data/durations/{data}.csv', delimiter=',')

    for _ in range(num_networks):
        k = claim_index(data)
        print(f'network {k} for data {data}', flush=True)
        samples_for_plot = []
        classifier = []
        samples = []
        for l, _ in enumerate(partitions):
            samples_for_plot.append([])
            classifier.append(sklearn.mixture.GaussianMixture(n_components=optimal_num_components[data][l], covariance_type='full'))
            egos_age = [a for a in egos if a['age'] == l]
            ## use log(k+1) instead of k to fit
            X = [[math.log(b+1) for b in a['contacts']] for a in egos_age]
            classifier[l].fit(X)
            ## sample same number of people as the data
            samples_tmp,_ = classifier[l].sample(per_partition[l])
            for sample in samples_tmp:
                samples.append([int(np.round(np.exp(b)-1)) if int(np.round(np.exp(b)-1))>=0 else 0 for b in sample])
                samples_for_plot[-1].append([int(np.round(np.exp(b)-1)) if int(np.round(np.exp(b)-1))>=0 else 0 for b in sample])
        res = nd_p.gmm_dur_gillesp_sc(samples,partitions=partitions,num_dur=3, taus=taus, iterations=48*2,props=props.tolist(), num_infec=1)
        with open(f'duration+ages/seir_sims/{data}_{k}{SUFFIX}_age_dur.json','w') as f:
            json.dump(res, f)
