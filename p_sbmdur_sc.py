# import sys
# import os
# sys.path.append(os.path.abspath('..'))
import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math

n = 100_000
num_networks= 50

# taus1 = np.linspace(0.073,.08,10)
# taus2 = np.linspace(.145,.155,10)
# taus3 = np.linspace(.25,.26,10)
# taus = np.concatenate([taus1, taus2, taus3])
taus = np.array([0.077,.15,.255])

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['poly']

model = 'sbm_dur'


def make_contact_matrices(egos, num_durs):
    num_per_bucket = np.zeros(np.max([a['age'] for a in egos])+1)
    contact_matrix = [np.zeros((np.max([a['age'] for a in egos])+1, np.max([a['age'] for a in egos])+1)) for _ in range(num_durs)]
    for ego in egos:
        num_per_bucket[ego['age']] += 1
        for j, val in enumerate(ego['contacts']):
            contact_matrix[j%num_durs][ego['age'], j//num_durs] += val
    for j in range(num_durs):
        contact_matrix[j] = np.divide(contact_matrix[j].T, num_per_bucket).T
        contact_matrix[j] = (contact_matrix[j] + contact_matrix[j].T)/2
    return contact_matrix, num_per_bucket

for i, data in enumerate(datas):
    with open(f'duration+ages/data/gmm_opt_comp/optimal_components_{data}_log_smalldur.json', 'r') as f:
        optimal_num_components = json.load(f)
    ##################### read fits ####################################
    with open(f'input_data/egos/{data}_dur_small.json', 'r') as f:
        egos = json.load(f)
    props = np.genfromtxt(f'input_data/durations/{data}.csv', delimiter=',')

    contact_matrix, num_per_bucket = make_contact_matrices(egos, num_durs=3)

    for k in range(num_networks):
        res = nd_p.sbm_gillesp_dur_sc(contact_matrix=contact_matrix, partitions=partitions, taus=taus, iterations=48*2, num_infec=1, props=props.tolist(), num_dur=3)
        with open(f'duration+ages/seir_sims/{data}_{k+50}_sbm_dur_age_dur.json','w') as f:
            json.dump(res, f)

