import nd_python_avon as nd_p 
import numpy as np
import pandas as pd
import json

n, iters = 100_000, 48

buckets = np.array([])
partitions = [n]


# datas = ['comix1','comix2','poly']
datas = ['comix1']
# datas = ['comix2']
# datas = ['poly']

models = ['dpln']

# scales = ['none']
scales = ['fit1']
# scales = ['fit2']

## 0,1
# taus = [[np.arange(0.001,0.07,0.002)]]
# taus = [[np.arange(0.001,0.07,0.002)]]
# taus = [[np.arange(0.001,0.05,0.002)]]
# taus = [[10*x for x in a] for a in taus]
## 2,3
# taus = [[np.arange(0.001,0.07,0.001)]]
# taus = [[np.arange(0.001,0.07,0.002)]]
# taus = [[np.arange(0.001,0.05,0.002)]]
# taus = [[20*x for x in a] for a in taus]
## 4,5
# taus = [[np.arange(0.005,0.4,0.005)]]
# taus = [[np.arange(0.001,0.07,0.002)]]
# taus = [[np.arange(0.001,0.05,0.002)]]
## 6,7
# taus = [[np.arange(0.4,0.605,0.005)]]
# taus = [[np.arange(0.07,0.09,0.002)]]
# taus = [[np.arange(0.05,0.09,0.002)]]
## 8,9,10,11,12
# taus = [[np.arange(0.005,0.205,0.005)]]
# taus = [[np.arange(0.001,0.02,0.002)]]
# taus = [[np.arange(0.001,0.04,0.002)]]
## 13
# taus = [[np.arange(0.4,0.805,0.005)]]
# taus = [[np.arange(0.07,0.11,0.002)]]
# taus = [[np.arange(0.05,0.11,0.002)]]
## 14,15
taus = [[np.arange(0.1,0.605,0.005)]]
# taus = [[np.arange(0.09,0.4,0.005)]]
# taus = [[np.arange(0.05,0.3,0.005)]]

# ## sc
# taus = [[[0.0965]]]
# # taus = [[[0.1094]]]
# # taus = [[[0.1932]]]

for i, data in enumerate(datas):
    for j, model in enumerate(models):
        print(data, model)
        # contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        # if model == 'sbm':
        #     params = []
        # else:
        #     params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        df = pd.read_csv(f'input_data/{data}.csv')
        _, contact_matrix, params = nd_p.fit_to_data(df=df,dist_type=model, buckets=buckets, save_fig=False)
        # result = {'zeros': [], 'avg_degree': [], 'max_degree': []}
        # for _ in range(100):
        #     network = nd_p.build_network(n, partitions, contact_matrix, params=params, dist_type=model)
        #     result['zeros'].append(len([a for a in network['degrees'] if a == 0]))
        #     result['avg_degree'].append(np.mean([a for a in network['degrees']]))
        #     result['max_degree'].append(max([a for a in network['degrees']]))
        result = nd_p.big_sellke_sims(partitions=partitions,contact_matrix=contact_matrix,network_params=params,n=n,dist_type=model,num_networks=1,iterations=iters, taus=taus[i][j],prop_infec=10/n, scaling=scales[j])
        with open(f'../output_data/simulations/big/sellke/SIR/30_{data}_{model}_{scales[j]}_noage.json','w') as f:
            json.dump(result, f)
print('done')
