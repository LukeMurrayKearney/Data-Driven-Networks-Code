# import sys
# import os
# sys.path.append(os.path.abspath('..'))
import nd_python_avon as nd_p 
import numpy as np
import json
import sklearn.mixture
import math
import matplotlib.pyplot as plt

n = 100_000
num_networks= 1

#taus = [np.linspace(0.1,1,10) for _ in range(4)]
taus = [[.45,.5,.55,2,2.5,15,16],
        [.5,.55,.6,11,14],
        [.6,.7,.75,12,14],
        [.055,.065,.075,.14,.15,.24,.26]]
        

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['comix3', 'comixa', 'comixb', 'poly']
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

    
    print(data)
    contact_matrix, num_per_bucket = make_contact_matrices(egos, num_durs=3)    
    # network_dur = nd_p.build_network(n=n, contact_matrix=contact_matrix, partitions=partitions, dist_type='sbm_dur', num_dur=3, props=props.tolist())
    # print(np.sum(np.mean(network_dur['degrees'], axis=0)), np.mean(network_dur['degrees'], axis=0))
    
    # print()
    # print(np.sum([np.tril(a, k=0) for a in contact_matrix]))
    # print(np.sum(np.tril(cm, k=0)))
    # print()
    res = {'taus': [], 'fs': [], 'I1': [], 'I2': [], 'I3': [], 'I4': [], 'peak_heights': [], 'peak_times': [], 'initial_infected': [], 'largest_connected_component': [], 'age_dur_sc': [], 'iteration': []}
    for k in range(num_networks):
        res_tmp = nd_p.sbm_gillesp_dur(contact_matrix=contact_matrix, num_dur=3, partitions=partitions, taus=taus[i], iterations=8, props=props.tolist(), num_infec=1)
        for tau, fs_row, I1_row, I2_row, I3_row, I4_row, peak_heights_row, peak_times_row, initial_infected_row, in zip(res_tmp["taus"], res_tmp["fs"], res_tmp['I1'], res_tmp['I2'], res_tmp['I3'], res_tmp['I4'], res_tmp['peak_heights'], res_tmp['peak_times'], res_tmp['initial_infected']):
            for fs, I1, I2, I3, I4, peak_height, peak_time, initial_infected in zip(fs_row, I1_row, I2_row, I3_row, I4_row, peak_heights_row, peak_times_row, initial_infected_row):
                if tau not in res['taus']:
                    res["taus"].append(tau)
                    res['iteration'].append(k)
                    res["fs"].append([fs])
                    res["I1"].append([I1])
                    res["I2"].append([I2])
                    res["I3"].append([I3])
                    res["I4"].append([I4])
                    res['peak_heights'].append([peak_height])
                    res['peak_times'].append([peak_time])
                    res['initial_infected'].append([initial_infected])
                else:
                    index = res["taus"].index(tau)
                    res["fs"][index].append(fs)
                    res["I1"][index].append(I1)
                    res["I2"][index].append(I2)
                    res["I3"][index].append(I3)
                    res["I4"][index].append(I4)
                    res['peak_heights'][index].append(peak_height)
                    res['peak_times'][index].append(peak_time)
                    res['initial_infected'][index].append(initial_infected)

    res = dict(res)  
    taus_plot = res["taus"]
    I1 = res["I1"]          
    I2 = res["I2"]           
    I3 = res["I3"] 
    I4 = res["I4"]  
    peak_heights = res['peak_heights']
    peak_times = res['peak_times']
    initial_infected =  res['initial_infected']
    largest_connected_component = res['largest_connected_component']
    iterations = res['iteration']

    fs = res['fs']

    sorted_data = sorted(
        zip(taus_plot, fs, I1, I2, I3, I4, peak_heights, peak_times, initial_infected, iterations),
        key=lambda x: x[0]
    )

    taus_plot, fs, I1, I2, I3, I4, peak_heights, peak_times, initial_infected, iterations = map(list, zip(*sorted_data))
    # sorted_data = sorted(zip(taus, fs, I1, I2, I3, I4, peak_heights, peak_times, initial_infected, largest_connected_component), key=lambda x: x)
    # taus, fs, I1, I2, I3, I4, peak_heights, peak_times, initial_infected, largest_connected_component = map(list, zip(*sorted_data))

    ##### choose r02 ########
    ## 1) R0<1 => R_inf=0
    ## 2) R0>1 => R_inf=mean(Zs|Zs>100)
    ## 3) Problem if R0>1 and Zs<100 for all simulations (probably wont happen)
    r0_mean = [np.sum(I2[idx])/np.sum(I1[idx]) if np.sum(I1[idx])!=0 else 0 for idx in range(len(I1))]
    r02_mean = [np.sum(I3[idx])/np.sum(I2[idx]) if np.sum(I2[idx])!=0 else 0 for idx in range(len(I1))]
    r03_mean = [np.sum(I4[idx])/np.sum(I3[idx]) if np.sum(I3[idx])!=0 else 0 for idx in range(len(I1))]
    fs_mean = [np.mean([b for b in a if b > 100/n]) if r02_mean[idx] > 1 else 0 for idx, a in enumerate(fs)]
    # print(model, data,r0_mean)

    plt.figure(figsize=(6,4))
    plt.scatter(taus_plot, r0_mean, marker='o', label="R₀ 1", alpha=0.2)
    plt.scatter(taus_plot, r02_mean, marker='o', label="R₀ 2", alpha=0.2)
    plt.scatter(taus_plot, r03_mean, marker='o', label="R₀ 3", alpha=0.2)
    plt.plot(taus_plot, r02_mean)

    plt.xlabel("τ")
    plt.ylabel("R₀ (mean across iterations)")
    plt.title(f"{data} - {model}: R₀ vs τ")
    # plt.ylim(0, 6.1)
    # plt.xlim(0,1)
    # plt.legend()
    plt.grid(True)
    plt.savefig(f'test_figs_{data}_{model}.png')
    # plt.show()

