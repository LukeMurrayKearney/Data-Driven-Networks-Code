import nd_python_avon as nd_p 
import numpy as np
import json

n, iters = 100_000, 30


buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

# datas = ['comix1','comix2','poly']
datas = ['comix1','comix2','comix3','poly']
models = ['sbm','dpln']
scales = ['none', 'none']

## 0
# taus = [[np.arange(0.001,0.1,0.0025), np.arange(0.001,0.06,0.001)],
#         [np.arange(0.001,0.1,0.0025), np.arange(0.001,0.06,0.001)],
#         [np.arange(0.001,0.1,0.0025), np.arange(0.001,0.1,0.0025)]]
## 1,2,3,4
# taus = [[np.arange(0.001,0.1,0.005), np.arange(0.001,0.06,0.002)],
#         [np.arange(0.001,0.1,0.005), np.arange(0.001,0.06,0.002)],
#         [np.arange(0.001,0.1,0.005), np.arange(0.001,0.1,0.005)]]
# 5,6,7,8,9
# taus = [[np.arange(0.1,0.205,0.005), np.arange(0.0005,0.01,0.0005)],
#         [np.arange(0.05,0.2,0.01), np.arange(0.0005,0.01,0.0005)],
#         [np.arange(0.001,0.1,0.005), np.arange(0.0025,0.05,0.0025)]]
# 10,11,12,13,14,15,16
# taus = [[np.arange(0.1,0.305,0.005), np.arange(0.0001,0.005,0.0001)],
#         [np.arange(0.05,0.25,0.005), np.arange(0.0001,0.005,0.0001)],
#         [np.arange(0.001,0.1,0.002), np.arange(0.0001,0.04,0.0001)]]
# 17
# taus = [[np.arange(0.28,0.38,0.005), np.arange(0.0001,0.005,0.0001)],
#         [np.arange(0.2,0.32,0.005), np.arange(0.0001,0.002,0.0001)],
#         [np.arange(0.001,0.08,0.001), np.arange(0.0001,0.03,0.0001)]]
# 18,19
# taus = [[np.arange(0.005,0.1,0.005), np.arange(0.0005,0.005,0.0005)],
#         [np.arange(0.005,0.1,0.005), np.arange(0.00005,0.001,0.00005)],
#         [np.arange(0.001,0.05,0.001), np.arange(0.0001,0.03,0.0001)]]
# 20,21,22
# taus = [[np.arange(0.005,0.05,0.005), np.arange(0.0001,0.005,0.0001)],
#         [np.arange(0.005,0.05,0.005), np.arange(0.00005,0.001,0.00005)],
#         [np.arange(0.001,0.025,0.001), np.arange(0.0005,0.015,0.0005)]]
# 23,24,25
# taus = [[np.arange(0.1,0.205,0.005), np.arange(0.0005,0.01,0.0005)],
#         [np.arange(0.05,0.2,0.01), np.arange(0.0005,0.01,0.0005)],
#         [np.arange(0.001,0.1,0.005), np.arange(0.0025,0.05,0.0025)]]
# 37
taus = [[np.arange(0.3,0.6,0.01), np.arange(0.001,0.01,0.001)],
        [np.arange(0.2,0.6,0.01), np.arange(0.001,0.01,0.001)],
        [np.arange(0.1,0.3,0.01), np.arange(0.005,0.05,0.005)]]

##comix3 1 ^ / switch to c2 for c3 from 6
taus = [[np.arange(0.1,0.205,0.005), np.arange(0.0005,0.01,0.0005)],
        [np.arange(0.05,0.2,0.01), np.arange(0.0005,0.01,0.0005)],
        [np.arange(0.001,0.1,0.005), np.arange(0.0025,0.05,0.0025)]]
## comix3 9 and up
taus = [[np.arange(0.01,0.5,0.01)]]

## 4
taus = [np.arange(0.3,2,0.1),
        np.arange(0.2,2,0.1),
        np.arange(0.4,2,0.1),
        np.arange(0.1,1,0.1)]



for i, data in enumerate(datas):
    for j, model in enumerate([models[0]]):
        print(data, model)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params = []
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        # result = {'zeros': [], 'avg_degree': [], 'max_degree': []}
        # for _ in range(100):
        #     network = nd_p.build_network(n, partitions, contact_matrix, params=params, dist_type=model)
        #     result['zeros'].append(len([a for a in network['degrees'] if a == 0]))
        #     result['avg_degree'].append(np.mean([a for a in network['degrees']]))
        #     result['max_degree'].append(max([a for a in network['degrees']]))
        # result = nd_p.build_network(n,partitions,contact_matrix,params,model)
        result = nd_p.big_sellke_sims(partitions=partitions,contact_matrix=contact_matrix,network_params=params,n=n,dist_type=model,num_networks=1,iterations=iters, taus=taus[i],prop_infec=10/n, scaling=scales[j])
        # print(sum(result['degrees']))
        with open(f'output_data/sims/54_{data}_{model}_{scales[j]}.json','w') as f:
            json.dump(result, f)
print('done')
