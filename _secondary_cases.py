import nd_python_avon as nd_p 
import numpy as np
import json
n, iters = 100_000, 50

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix1','comix2','poly']
models = ['sbm','dpln']
scales = ['none', 'none']

## 0
# taus = [[[0.07,0.117,0.154], [0.00036,0.00045,0.00054]],
#         [[0.054,0.08,0.11] , [0.00027,0.00036,0.00046]],
#         [[0.024,0.03,0.038], [0.014  , 0.0185,  0.023]]]
## last
# taus = [[[0.033], [0.0002]],
#         [[0.036], [0.00016]],
#         [[0.04], [0.023]]]
## lasty
taus = [[[0.032], [0.00017]],
        [[0.0358], [0.000155]],
        [[0.0395], [0.0225]]]

for i, data in enumerate(datas):
    for j, model in enumerate(models):
        if i!=0 and j!=1:
            continue
        print(data, model)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params = []
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        result = nd_p.big_sellke_sims(partitions=partitions,contact_matrix=contact_matrix,network_params=params,n=n,dist_type=model,num_networks=30,iterations=iters, taus=taus[i][j],prop_infec=10/n, scaling=scales[j], secondary_cases=True)
        with open(f'../output_data/simulations/big/sellke/SIR/last_12_{data}_{model}_{scales[j]}_sc.json','w') as f:
            json.dump(result, f)
print('done')
