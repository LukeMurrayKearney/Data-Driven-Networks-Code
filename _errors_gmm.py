import nd_python as nd_p 
import numpy as np
import csv
import sys
import sklearn
import json
import math

data = 'comix1'
model = 'gmm'

n, iters = 100_000, 30

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]
per_partition = [a+1 if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]


with open(f'input_data/gmm/optimal_components_{data}_log.json', 'r') as f:
    optimal_num_components = json.load(f)
##################### read fits ####################################
with open(f'input_data/egos/{data}.json', 'r') as f:
    egos = json.load(f)
print(data, model)

# datas = ['poly','comix1', 'comix2']
# models = ['sbm', 'nbinom', 'dpln']

# distance matrix of EMD
bins = np.arange(0,len(partitions),1)
distance_matrix = np.zeros((len(partitions), len(partitions)))
for i in bins:
    for j in bins:
        distance_matrix[i,j] = np.float64(np.abs(i-j))

print(data, model)
error, error_breakdown = [], []
error_with_itself, error_with_itself_breakdown = [], []
for i in range(iters):
    # my model error
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
    network = nd_p.build_gmm(degree_dist=samples, partitions=partitions)
    errors, err_pp = nd_p.emd_error(egos, network, distance_matrix=distance_matrix)
    error_breakdown.append(errors)
    error.append(err_pp)
    
    # network error of my model with true network
    new_data = nd_p.data_from_network(network=network)
    egos_itself, contact_matrix_itself, params_itself = nd_p.fit_to_data(df=new_data, save_fig=False, output_file_path="fits/network_comix1", buckets=buckets,dist_type=model)
    network = nd_p.build_network(n=n, partitions=partitions, params=params_itself, contact_matrix=contact_matrix_itself,dist_type=model)
    
    if i % 1 == 0:
        print(i)
    errors, err_pp = nd_p.emd_error(egos=egos_itself, network=network, distance_matrix=distance_matrix)
    error_with_itself_breakdown.append(errors)
    error_with_itself.append(err_pp)
    
with open(f'output_data/errors/breakdown_{data}_{model}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in error_breakdown:
        writer.writerow(row)
        
with open(f'output_data/errors/breakdown_itself_{data}_{model}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in error_with_itself_breakdown:
        writer.writerow(row)

with open(f'output_data/errors/{data}_{model}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(error)


with open(f'output_data/errors/itself_{data}_{model}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(error_with_itself)

print(f'done: {data} {model}')