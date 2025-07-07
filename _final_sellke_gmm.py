import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math

n, iters = 100_000, 30
num_networks = 1

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a+1 if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

# datas = ['comix1','comix2','poly']
datas = ['comix1','comix2','comix3','poly']
models = ['gmm']
scales = ['none']

## 0,1
taus = [np.arange(0.001,0.03,0.001),
        np.arange(0.001,0.03,0.001),
        np.arange(0.005,0.11,0.005)]
## 2 / comix3 0, used c2 from 85
taus = [np.arange(0.0005,0.007,0.0005),
        np.arange(0.0005,0.005,0.0005),
        np.arange(0.002,0.07,0.002)]
#comix3 1
taus = [np.arange(0.005,0.1,0.005)]

## 4 
taus = [np.arange(0.02,0.52,0.02),
        np.arange(0.02,0.52,0.02),
        np.arange(0.02,0.52,0.02),
        np.arange(0.02,0.52,0.02)]

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
                # print(samples)
            print(scales[0])
            print(taus[i])
            result = nd_p.gmm_sims(samples,partitions=partitions,taus=taus[1], iterations=iters, inv_gamma=7, prop_infec=10/n, scaling=scales[0])
                
            with open(f'output_data/gmm/4_{k+21}_{data}_{model}_{scales[j]}.json','w') as f:
                json.dump(result, f)
print('done')
