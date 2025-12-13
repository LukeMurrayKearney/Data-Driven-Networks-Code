import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math

n = 100_000
num_networks= 20

taus2 = np.arange(0.006, 0.55, 0.025)
taus3 = np.arange(0.55, 3.5, 0.1)
taus1 = np.arange(3.5, 12.5, 1)
taus = np.concatenate((taus2, taus3, taus1))

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['comixb']


for i, data in enumerate(datas):
    with open(f'input_data/gmm/optimal_components_{data}_log.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}.json', 'r') as f:
        egos = json.load(f)
    cm = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')

    for k in range(num_networks):
        res = nd_p.sbm_gillesp(contact_matrix=cm, partitions=partitions, taus=taus, iterations=48, num_infec=1)
        with open(f'duration+ages/seir_sims/{data}_{k+120}_sbm_fin.json','w') as f:
            json.dump(res, f)
