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

# taus2 = np.arange(.00025, .006, .0005)
# taus1 = np.arange(.006, .055, .002)
# # taus3 = np.arange(.055, 1.5, .1)
# taus = np.concatenate((taus2, taus1))

## taus placing 24 points evenly across R0 = 0.8-5.5, by inverting the
## measured tau -> R0 curve of the runs already on disk; see the tau
## coverage section of thesis_figs.ipynb
taus = np.array([0.0027928, 0.0037712, 0.0045347, 0.0054322, 0.0058842, 0.0072917, 0.00777, 0.0084599, 0.0095786, 0.010413, 0.011026, 0.012168, 0.012715, 0.014056, 0.014552, 0.015144, 0.016265, 0.018147, 0.01871, 0.020223, 0.021018, 0.021641, 0.022315, 0.023091])


# taus = np.arange(0.05, .12, 0.005)

buckets = np.array([])
partitions = [n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['reconnect']


SUFFIX = '_nodur_noage'   # duration+ages/seir_sims/{data}_{k}{SUFFIX}_fin.json


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
    with open(f'input_data/gmm/optimal_components_{data}_log_noage.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}_noage.json', 'r') as f:
        egos = json.load(f)

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
        res = nd_p.gmm_gillesp(samples,partitions=partitions,taus=taus, iterations=48, num_infec=1)
        with open(f'duration+ages/seir_sims/{data}_{k}{SUFFIX}_fin.json','w') as f:
            json.dump(res, f)
