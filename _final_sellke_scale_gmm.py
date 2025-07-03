import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math

n, iters = 100_000, 48
num_networks = 20

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a+1 if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]


# datas = ['comix1','comix2','poly']
datas = ['comix1','comix3']
models = ['gmm']
scales = ['fit1', 'fit2']

## 0
# taus = [np.arange(0.0005,0.02,0.0005),
#         np.arange(0.0005,0.15,0.0005),
#         np.arange(0.0005,0.10,0.0005)]
# taus = [[15*x for x in a] for a in taus]

# ## 1 Need to make max larger for scaled (need to move up to 5)
# taus = [np.arange(0.0005,0.02,0.0005),
#         np.arange(0.0005,0.15,0.0005),
#         np.arange(0.0005,0.10,0.0005)]
# taus = [[20*x for x in a] for a in taus]

## 2 / comix3 0, 85+ used c2 for c3
taus = [np.arange(0.01,0.4,0.01),
        np.arange(0.01,0.6,0.01),
        np.arange(0.01,0.5,0.01)]

# 3,1 comix 1 and comix 3
taus = [np.arange(0.35,0.6,0.01),
        np.arange(0.3,0.7,0.01)]


for i, data in enumerate(datas):
    for j, model in enumerate(models):
        with open(f'input_data/gmm/optimal_components_{data}_log.json', 'r') as f:
            optimal_num_components = json.load(f)
        ##################### read fits ####################################
        with open(f'input_data/egos/{data}.json', 'r') as f:
            egos = json.load(f)
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
            
            result = nd_p.gmm_sims(samples,partitions=partitions,taus=taus[j], iterations=iters, inv_gamma=7, prop_infec=10/n, scaling=scales[j])
                
            with open(f'output_data/gmm/{3 if j==0 else 1}_{k}_{data}_{model}_scale.json','w') as f:
                json.dump(result, f)
print('done')