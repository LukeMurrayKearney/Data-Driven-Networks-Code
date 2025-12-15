# import sys
# import os
# sys.path.append(os.path.abspath('..'))
import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math

n = 100_000
num_networks= 20

taus1 = np.arange(.02, 1.0, .02)
taus2 = np.arange(1.0, 4, .05)
taus = np.concatenate((taus1, taus2))

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['comix3']


for i, data in enumerate(datas):
    with open(f'duration+ages/data/gmm_opt_comp/optimal_components_{data}_log_smalldur.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}_dur_small.json', 'r') as f:
        egos = json.load(f)
    props = np.genfromtxt(f'input_data/durations/{data}.csv', delimiter=',')

    for k in range(num_networks):
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
        res = nd_p.gmm_dur_gillesp(samples,partitions=partitions,num_dur=3, taus=taus, iterations=48,props=props.tolist(), num_infec=1)
        with open(f'duration+ages/seir_sims/{data}_{k+160}_fin.json','w') as f:
            json.dump(res, f)
