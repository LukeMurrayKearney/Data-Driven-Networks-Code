import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math

n = 100_000
num_networks= 40

taus1 = np.arange(.0015, .08, .001)

buckets = np.array([])
partitions = [n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['comixb']


for i, data in enumerate(datas):
    with open(f'input_data/gmm/optimal_components_{data}_log_noage.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}_noage.json', 'r') as f:
        egos = json.load(f)

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
        res = nd_p.gmm_gillesp(samples,partitions=partitions,taus=taus1, iterations=48, num_infec=1)
        with open(f'duration+ages/seir_sims/{data}_{k+200}_nodur_noage_fin.json','w') as f:
            json.dump(res, f)