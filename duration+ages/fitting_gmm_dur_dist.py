import sys
import os
sys.path.append(os.path.abspath('..'))
import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math
import matplotlib.pyplot as plt

n = 100_000
max_comps = 20
buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a+1 if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-5 mins', '1-4 hours', '4+ hours']
datas = ['comixa']
optimal = {}
for i, data in enumerate(datas):
    # ##################### read fits ####################################
    with open(f'../input_data/egos/{data}_dur.json', 'r') as f:
        egos = json.load(f)
    ## change egos from 5 durations to 3. (0-1,1-4,4+)
    egos_new = []
    for ego in egos:
        egos_new.append({'age': ego['age'], 'contacts': [a if index%5 in [3,4] else ego['contacts'][index]+ego['contacts'][index+1]+ego['contacts'][index+2] for index, a in enumerate(ego['contacts']) if index%5 in [0,3,4]], 'degree': ego['degree']})
    with open(f'../input_data/egos/{data}_dur_small.json','w') as f:
        json.dump(egos_new, f)
    for model_num in range(3):
        if model_num != 1: 
            continue
        with open(f'../input_data/egos/{data}_dur_small.json','r') as f:
            egos = json.load(f)
        for age, _ in enumerate(partitions):
            print(age)
            egos_age = [a for a in egos if a['age'] == age]
            ## use log(k+1) instead of k to fit
            X = [[b if model_num == 0 else math.log(b+1) if model_num == 1 else math.sqrt(0 if math.log(b+1) < 0 else math.log(b+1)) for b in a['contacts']] for a in egos_age]
            ## split into training and testing data 
            # Generate random indices
            indices = np.arange(len(X))
            split_idx = int(0.8 * len(X))

            bic_scores = []

            # find optimal number of components
            for i in range(1,max_comps):
                print(i)
                bic_scores.append([])
                for j in range(10):
                    np.random.shuffle(indices)

                    # Split data
                    X_train = [X[i] for i in indices[:split_idx]]
                    X_test = [X[i] for i in indices[split_idx:]]
                    bic_scores[-1].append([0,0])
                    classifier = sklearn.mixture.GaussianMixture(n_components=i, covariance_type='full')
                    classifier.fit(X_train)
                    bic_scores[-1][-1][0] = classifier.bic(np.array(X_train))/4
                    bic_scores[-1][-1][1] = classifier.bic(np.array(X_test))
                
            with open(f'data/gmm_bic/{data}_BIC_{age}_{"none" if model_num==0 else "log" if model_num==1 else "sqrt"}_smalldur.json', 'w') as f:
                json.dump(bic_scores, f)
            # try: 
            with open(f'data/gmm_bic/{data}_BIC_{age}_{"none" if model_num==0 else "log" if model_num==1 else "sqrt"}_smalldur.json', 'r') as f:
                bic_scores = json.load(f)

    for model_num in range(3):
        if model_num != 1: 
            continue
        print(data, model_num)
        optimal[data] = []
        ################ plot optimal number of components ###############
        for age, _ in enumerate(partitions):
            with open(f'data/gmm_bic/{data}_BIC_{age}_{"none" if model_num==0 else "log" if model_num==1 else "sqrt"}_smalldur.json', 'r') as f:
                bic_scores = json.load(f)
            test_vals = [np.median([b[1] for b in a]) for a in bic_scores]
            optimal[data].append(int(np.argmin(test_vals))+1)
        with open(f'data/gmm_opt_comp/optimal_components_{data}_{"none" if model_num==0 else "log" if model_num==1 else "sqrt"}_smalldur.json', 'w') as f:
            json.dump(optimal, f)
        print(optimal)