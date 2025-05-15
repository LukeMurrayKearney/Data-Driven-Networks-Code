import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math

n, iters = 50_000, 48
num_networks = 5

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a+1 if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

datas = ['comix1','comix2','poly']
models = ['gmm']
scales = ['none', 'none']

## 0
taus = [[[0.00017]],
        [[0.000155]],
        [[0.0225]]]

for i, data in enumerate(datas):
    with open(f'input_data/gmm/optimal_components_{data}_log.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}.json', 'r') as f:
        egos = json.load(f)
    for j, model in enumerate(models):
        print(data, model)
        for k in range(num_networks):
            classifier = []
            samples = []
            for l, _ in enumerate(partitions):
                classifier.append(sklearn.mixture.GaussianMixture(n_components=optimal_num_components[data][l], covariance_type='full'))
                egos_age = [a for a in egos if a['age'] == l]
                ## use log(k+1) instead of k to fit
                X = [[math.log(b+1) for b in a['contacts']] for a in egos_age]
                classifier[l].fit(X)
                ## sample same number of people as the data
                samples_tmp,_ = classifier[l].sample(per_partition[l])
                for sample in samples_tmp:
                    samples.append([int(np.round(np.exp(b)-1)) if int(np.round(np.exp(b)-1))>=0 else 0 for b in sample])
                # samples.append([[np.round(np.exp(b)-1) if np.round(np.exp(b)-1)>=0 else 0 for b in a] for a in samples_tmp])
                # print(np.shape(samples))
            
            result = nd_p.gmm_sims_sc(samples,partitions=partitions,taus=taus[i][j], iterations=iters, inv_gamma=7, prop_infec=10/n, scaling=scales[j])
                
            with open(f'output_data/gmm/0_{k}_{data}_{model}_sc.json','w') as f:
                json.dump(result, f)
print('done')
