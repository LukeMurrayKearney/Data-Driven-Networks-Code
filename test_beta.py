# import sys
# import os
# sys.path.append(os.path.abspath('..'))
import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math
import matplotlib.pyplot as plt

n = 10_000
num_networks= 5

taus = [np.linspace(0.1,1,10) for _ in range(4)]

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['comix3', 'comixa', 'comixb', 'poly']


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

    
    print(data)
    contact_matrix, num_per_bucket = make_contact_matrices(egos, num_durs=3)    
    network_dur = nd_p.build_network(n=n, contact_matrix=contact_matrix, partitions=partitions, dist_type='sbm_dur', num_dur=3, props=props.tolist())
    print(np.sum(np.mean(network_dur['degrees'], axis=0)), np.mean(network_dur['degrees'], axis=0))
    
    with open(f'input_data/egos/{data}.json', 'r') as f:
        egos_noage = json.load(f)
    cm = make_contact_matrices(egos_noage, num_durs=1)[0][0]
    
    # cm = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
    network = nd_p.build_network(n=n, contact_matrix=cm, partitions=partitions, dist_type='sbm')
    print(np.mean(network['degrees'], axis=0))
    
    # print()
    # print(np.sum([np.tril(a, k=0) for a in contact_matrix]))
    # print(np.sum(np.tril(cm, k=0)))
    # print()
    # for k in range(num_networks):

        #   res = nd_p.sbm_gillesp_dur(contact_matrix=contact_matrix, num_dur=3, partitions=partitions, taus=taus[i], iterations=10, props=props.tolist(), num_infec=1)
    #     print(data)
    #     I2, I3 = res['I2'], res['I3']
    #     r0s = [0 if np.sum(I2[int(tau)])<=0 else np.sum(I3[int(tau)])/np.sum(I2[int(tau)]) for tau,_ in enumerate(taus[i])]
    #     plt.scatter(taus[i], r0s)
    #     plt.plot(taus[i], r0s, label=f'{k}')
    # plt.title(f'{data}')
    # plt.show()
