import nd_rust as nd_r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sc
import math
import itertools
from multiprocessing import Pool
import random
# import pyemd
import json
from scipy.stats import nbinom, poisson, geom
from scipy.optimize import minimize
plt.rcParams.update({'font.size': 14})  # Adjust the font size

################################## build into a package ##################################
def big_sellke_sims(partitions, contact_matrix, taus=np.arange(0.1,1,0.1), num_networks=5, iterations=10, network_params=None, n=100_000, dist_type='nbinom', inv_gamma=7, prop_infec=1e-3, scaling="None", secondary_cases=False):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0,inv_gamma]
    if secondary_cases:
        return nd_r.big_sellke_sec_cases(taus, num_networks, iterations, n, partitions, dist_type, network_params, contact_matrix, outbreak_params, prop_infec, scaling)
    else:
        return nd_r.big_sellke(taus, num_networks, iterations, n, partitions, dist_type, network_params, contact_matrix, outbreak_params, prop_infec, scaling)

def sellke_sims_growth_rate(partitions, contact_matrix, taus=np.arange(0.1,1,0.1), num_networks=5, iterations=10, network_params=None, n=100_000, dist_type='nbinom', inv_gamma=7, prop_infec=1e-3, scaling="None"):
    partitions = [int(a) for a in partitions]
    outbreak_params = [0,inv_gamma]
    
    return nd_r.big_sellke_growth_rate(taus, num_networks, iterations, n, partitions, dist_type, network_params, contact_matrix, outbreak_params, prop_infec, scaling)


def sellke_test(partitions, contact_matrix, network_params=None, tau=0.25, iterations=1, n=100_000, dist_type='nbinom', inv_gamma=7, prop_infec=1e-3, scaling="None"):
    partitions = [int(a) for a in partitions]
    parameters = [tau,inv_gamma]
    # if iterations == 1:
    return nd_r.sellke_sim(iterations=iterations, n=n, partitions=partitions, dist_type=dist_type, network_params=network_params, contact_matrix=contact_matrix, outbreak_params=parameters, prop_infec=prop_infec, scaling=scaling)


def mcmc(data, days, partitions, contact_matrix, network_params, outbreak_params, tau_0=0.02, p_hosp=0.01, iters=10_000, dist_type='nbinom', n=20_000, prior_param=5,scaling='fit1'):
    partitions = [int(a) for a in partitions]
    return nd_r.mcmc_data(data=data, days=days, tau_0=tau_0, proportion_hosp=p_hosp, iters=iters, dist_type=dist_type, n=n, partitions=partitions, contact_matrix=contact_matrix, network_params=network_params, outbreak_params=outbreak_params, prior_param=prior_param,scaling=scaling)

def fit_to_data(df = None, input_file_path = 'input_data/poly.csv', dist_type = "nbinom", buckets = np.array([5,12,18,30,40,50,60,70]), save_fig = True, output_file_path=None, log=False, to_csv=False, fig_data_file='',num_bins=15):

    # Call the function with the provided arguments
    if df is None:
        df = read_in_dataframe(input_file_path)
    
    # Create list of ego networks from data
    egos = make_egos_list(df=df, buckets=buckets)
    
    # Create Contact Matrices
    contact_matrix, num_per_bucket = make_contact_matrices(egos=egos, buckets=buckets)
    
    # fit to the ego networks for each age bracket
    params = fit_dist(egos, dist_type, buckets, num_per_bucket, save_fig=save_fig, file_path=output_file_path, log=log,to_csv=to_csv,fig_data_file=fig_data_file,num_bins=num_bins)

    # print("Fitting complete.")
    return egos, contact_matrix, params

def build_network(n, partitions, contact_matrix, params=None, dist_type ="nbinom"):
    partitions = [int(a) for a in partitions]
    if dist_type == 'sbm':
        network = nd_r.sbm_from_vars(n, partitions, contact_matrix)
    else:
        if params is None:
            print("Parameters are required")
        network = nd_r.network_from_vars(n, partitions, dist_type, params, contact_matrix)
    return network

def to_networkx(network={}):
    G = nx.Graph()
    G.add_nodes_from(range(len(network['ages'])))
    nx.set_node_attributes(G,network['ages'], 'age')
    for person in network['adjacency_matrix']:
        for link in person:
            G.add_edge(link[0], link[1])
    return G    

def fit_to_r0(partitions, contact_matrix, r0=3, network_params=None, iterations=30, n=30_000, dist_type='nbinom', inv_gamma=7, prop_infec=1e-3, num_networks=30, num_restarts=30, scaling="None"):
    
    outbreak_params = [r0, inv_gamma]
    partitions = [int(a) for a in partitions]
    if dist_type == 'sbm':
        network_params = []
    return nd_r.test_r0_fit(n=n, partitions=partitions, dist_type=dist_type, network_params=network_params, contact_matrix=contact_matrix, outbreak_params=outbreak_params, prop_infec=prop_infec, num_networks=num_networks, target_r0=r0, iters=iterations, num_replays=num_restarts, scaling=scaling)

def simulate(partitions, contact_matrix, network_params=None, tau=0.05, iterations=50, n=30_000, dist_type='nbinom', maxtime=10_000, inv_gamma=7, prop_infec=1e-3, scaling="None"):
    ## Need to write some sensible code for this, depending on if we want to give params or a network
    partitions = [int(a) for a in partitions]
    parameters = [tau,inv_gamma]
    return nd_r.infection_sims(iters=iterations, n=n, partitions=partitions, dist_type=dist_type, network_params=network_params, contact_matrix=contact_matrix, outbreak_params=parameters, maxtime=maxtime, prop_infec=prop_infec, scaling=scaling)
    
def taus_sims(partitions, contact_matrix, taus=[0.05], network_params=None, iterations=50, n=30_000, dist_type='nbinom', maxtime=10_000, inv_gamma=7, prop_infec=1e-3, scaling="None"):
    ## Need to write some sensible code for this, depending on if we want to give params or a network
    partitions = [int(a) for a in partitions]
    parameters = [0.05,inv_gamma]
    return nd_r.big_sims(taus=taus, iters=iterations, n=n, partitions=partitions, dist_type=dist_type, network_params=network_params, contact_matrix=contact_matrix, outbreak_params=parameters, maxtime=maxtime, prop_infec=prop_infec, scaling=scaling)

# def emd_error(egos, network, distance_matrix = None, extra_mass_penalty = 5):
#     # check distance matrix and number per bucket, if empty revert to default checks
#     if distance_matrix is None:
#         num_buckets = network['ages'][-1] + 1
#         bins = np.arange(0,num_buckets,1) 
#         distance_matrix = np.zeros((num_buckets, num_buckets))
#         for i in bins:
#             for j in bins:
#                 distance_matrix[i,j] = np.float64(np.abs(i-j))

#     num_buckets = network['ages'][-1] + 1
#     num_per_bucket = np.zeros(num_buckets)
#     for ego in egos:
#         num_per_bucket[ego['age']] += 1
    
#     errors, errors_pp = calc_error(egos=egos, network=network, distance_matrix=distance_matrix, extra_mass_penalty=extra_mass_penalty, num_per_bucket=np.array(num_per_bucket, dtype=int))
 
#     return errors, errors_pp

def data_from_network(network, n = 10_000, buckets = np.array([5,12,18,30,40,50,60,70])):
    
    # choose n random people from the network to create the data set 
    indices = []
    num_per_bucket = np.round(n * np.diff(np.array([0] + network['partitions'])) / network['partitions'][-1]).astype(int)
    for i, top in enumerate(network['partitions']):
        indices.append(random.sample(range(top), num_per_bucket[i]) if i == 0 else random.sample(range(network['partitions'][i-1], top), num_per_bucket[i]))
    network_data = [[np.array(network['frequency_distribution'][idx], dtype=int) for idx in age_class] for age_class in indices]

    # convert these ego networks to dataframe rows
    df = pd.DataFrame({
        'part_id': [],
        'cont_id': [],
        'cnt_age_exact': [],
        'part_age': [],
    })
    part_id, cont_id = 0, 0
    buckets = np.insert(buckets, 0, 0)
    buckets = np.insert(buckets, len(buckets), 120)
    for age_ego, age_class in enumerate(network_data):
        for ego in age_class:
            part_id += 1
            for age_cont, num_contacts in enumerate(ego):
                for _ in range(num_contacts):
                    cont_id += 1
                    # add each contact from ego network
                    df.loc[len(df)] = [part_id, cont_id, np.random.randint(buckets[age_cont], buckets[age_cont+1]), np.random.randint(buckets[age_ego], buckets[age_ego+1])]
            # check if individual has no contacts
            if sum(ego) == 0:
                df.loc[len(df)] = [part_id, None, None, np.random.randint(buckets[age_ego], buckets[age_ego+1])]
    return df


################################################# utils ########################################################

def read_in_dataframe(file_path):
    df = pd.read_csv(file_path)
    # remove NaNs
    df = df[df['part_age'].notna()]
    # remove values where ages not given
    rows_to_remove = df[(df['cnt_age_exact'].isna()) & (df['cont_id'].notna())].index
    df.drop(index=rows_to_remove, inplace=True)
    sorted_df = df.sort_values(by='part_id', ascending=False)
    return sorted_df

# Function to determine the bucket index for a given number
def get_bucket_index(num, buckets):
    for i, max_value in enumerate(buckets):
        if num < max_value:
            return i
    return len(buckets)  # If the number exceeds the last bucket, put it in the last bucket

def make_contact_matrices(egos,buckets):
    num_per_bucket = np.zeros(len(buckets)+1)
    contact_matrix = np.zeros((len(buckets)+1, len(buckets)+1))
    for ego in egos:
        num_per_bucket[ego['age']] += 1
        for j, val in enumerate(ego['contacts']):
            contact_matrix[ego['age'], j] += val
    contact_matrix = np.divide(contact_matrix.T, num_per_bucket).T
    return contact_matrix, num_per_bucket

# def make_contact_matrices(df, buckets):
#     num_per_bucket = np.zeros(len(buckets)+1)
#     contact_matrix = np.zeros((len(buckets)+1, len(buckets)+1))
#     # save last participant id
#     last_id = ''
#     # Iterate through the DataFrame and update the count_matrix
#     for _, row in df.iterrows():
#         b_i = get_bucket_index(row['part_age'], buckets=buckets)
#         if last_id != row['part_id']:
#             # count new participants
#             num_per_bucket[b_i] += 1
#         if pd.isnull(row['cont_id']) or pd.isnull(row['cnt_age_exact']):
#             continue
#         b_j = get_bucket_index(row['cnt_age_exact'], buckets=buckets)
#         contact_matrix[b_i, b_j] += 1
#         contact_matrix[b_j, b_i] += 1
#         last_id = row['part_id']
#     contact_matrix = np.divide(contact_matrix.T, num_per_bucket).T
#     return contact_matrix, num_per_bucket

def make_egos_list(df, buckets):
    egos = []
    last = ''
    # iterate through each contact
    for _, x in df.iterrows():
        if x['part_id'] == last:
            if np.isnan(x['cnt_age_exact']):
                continue
            else:
                j = get_bucket_index(x['cnt_age_exact'], buckets=buckets)
                egos[-1]['contacts'][j] += 1
        else:
            i = get_bucket_index(x['part_age'], buckets=buckets)
            egos.append({'age': i, 'contacts': np.zeros(len(buckets) + 1), 'degree': 0})
            if np.isnan(x['cnt_age_exact']):
                continue
            else:
                j = get_bucket_index(x['cnt_age_exact'], buckets=buckets)
                egos[-1]['contacts'][j] += 1
        last = x['part_id']

    # count degree of each node
    for i, _ in enumerate(egos):
        egos[i]['degree'] = np.sum(egos[i]['contacts'])
    # sort the egos by age group
    egos = sorted(egos, key=lambda x: x['age'])
    
    return egos

def fit_dist(egos, dist_type, buckets, num_per_bucket, save_fig=False, file_path=None, log=False, to_csv=False, fig_data_file='', num_bins=15):
    params = []
    # plotting
    num_subplots = len(num_per_bucket)
    num_cols = int(math.ceil(math.sqrt(num_subplots)))
    num_rows = int(math.ceil(num_subplots / num_cols))
    if save_fig:
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(8*num_rows,5*num_cols), constrained_layout=True)
        ax = ax.flatten()
        
    # Calculate cumulative sums
    cumulative_num_per = list(itertools.accumulate(num_per_bucket))
    
    if dist_type == "sbm":
        return None
          
    elif dist_type == "dpln":
        for i, num in enumerate(cumulative_num_per):
            if i == 0: 
                contacts = egos[0:int(num)]
            else: 
                contacts = egos[int(cumulative_num_per[i-1]):int(num)]
            # minimise negative log likelihood to find distribution fit to the data
            contacts = [a['degree']+1 for a in contacts]
            x = np.log(np.array(contacts))
            prior_params = [1, 1, 0, 4, 0.5, 0.5]
            result = nd_r.fit_dpln(x, 200_000, prior_params)
            parameters = [np.mean(result['alpha'][1_000:]), np.mean(result['beta'][1_000:]), np.mean(result['nu'][1_000:]), np.mean(result['tau'][1_000:])]
            params.append(parameters)
            if save_fig:
                plot_degree_dist_fit(contacts, buckets, params=params[-1], idx=i, ax=ax, dist_type=dist_type, num_rows=num_rows, num_cols=num_cols,log=log,to_csv=to_csv, graph_file=fig_data_file, num_bins=num_bins)
              
    else:
        for i, num in enumerate(cumulative_num_per):
            if i == 0: 
                contacts = egos[0:int(num)]
            else: 
                contacts = egos[int(cumulative_num_per[i-1]):int(num)]
            initial_r, initial_p = 1, 0.5
            # minimise negative log likelihood to find distribution fit to the data
            result = minimize(log_likelihood_nbinom, x0=[initial_r, initial_p], args=(contacts), method='Nelder-Mead')
            estimated_r, estimated_p = result.x
            params.append([estimated_r, estimated_p])
            if save_fig:
                plot_degree_dist_fit(contacts, buckets, params=params[-1], idx=i, ax=ax, dist_type=dist_type, num_rows=num_rows, num_cols=num_cols,log=log,to_csv=to_csv, graph_file=fig_data_file, num_bins=num_bins)
        
    
    if save_fig:
        if file_path is None:
            plt.savefig(dist_type + "_fits.png")
        else:
            plt.savefig(file_path)
    # else:
    #     plt.show()
    
    # change the form of the list of parameters to be read by rust
    if dist_type != "power_law":
        separated_lists = zip(*params)
        params = [list(group) for group in separated_lists]
    else:
        params = [list(a) for a in params]

    return params

######################################### Negative Binomial ###########################################

def log_likelihood_nbinom(params, egos):
    r, p = params
    contacts = [a['degree'] for a in egos]
    log_likelihood = nbinom.logpmf(contacts, r, p)
    return -np.sum(log_likelihood)


def plot_degree_dist_fit(contacts, buckets, params = None, idx = 0, ax = plt.gca(), dist_type = "nbinom", num_rows=1, num_cols=1, log=False, num_bins=15, to_csv=False,graph_file=''):
    
    if dist_type != "dpln":
        contacts = [a['degree'] for a in contacts]
    if len(contacts) == 0:
        print(f'No data points in age group: {idx}')
        return

    unique = np.unique(np.array(contacts)+1,return_counts=True)
    # plotting data
    if log == False:
        xs,ys = unique[0], unique[1]/sum(unique[1])
        ax[idx].scatter(unique[0], unique[1]/sum(unique[1]), label="Data")
        if to_csv == True:
            np.savetxt(f'graph_data/data_{graph_file}{idx}.csv', np.array([xs,ys]), delimiter=',')
    else:
        xs, ys = log_bins(x=contacts, num_bins=num_bins)
        xs, ys = [x for i, x in enumerate(xs) if ys[i] > 0], [y for y in ys if y > 0]
        ax[idx].scatter(xs, ys, label="Data")
        if to_csv == True:
            np.savetxt(f'graph_data/data_{graph_file}{idx}.csv', np.array([xs,ys]), delimiter=',')
    
    if dist_type == "dpln":
        if params != None or params != (0,0,0,0):
            # xs = np.logspace(min(np.log10(contacts)),max(np.log10(contacts)), 1000)
            x = np.arange(0.5, max(contacts)+10,0.1)
            y = nd_r.dpln_pdf(x, params)
            ax[idx].plot(x, y, 'r', lw=1,label="Fitted dPlN Distribution")
            if to_csv == True:
                np.savetxt(f'graph_data/fit_{dist_type}{idx}.csv', np.array([x,y]), delimiter=',')
    else: #dist_type == "nbinom":
        if params != None or params != (0,0):
            max_x = np.max(contacts)
            x = np.arange(0, max_x+10)
            pmf_nbinom = nbinom.pmf(x, params[0], params[1])
            ax[idx].plot(x+1, pmf_nbinom, 'ro-', lw=0.5,label="Fitted Negative Binomial")
            if to_csv == True:
                np.savetxt(f'graph_data/fit_{graph_file}{idx}.csv', np.array([x+1,pmf_nbinom]), delimiter=',')
         
    # else:
    #     if params != None or params != (0,0,0):
    #         max_x = np.max(contacts)
    #         x = np.arange(0, max_x + 1)
    #         pmf_pois = poisson.pmf(x, params[0])*params[2]
    #         pmf_geom = geom.pmf(x,params[1])*(1-params[2])
    #         ax[idx].plot(x+1, pmf_geom+pmf_pois, 'ro-', lw=0.5,label="Fitted Poisson-Geometric Mixture")
    if log == False:
        ax[idx].set_ylim([min(unique[1]/sum(unique[1]))/2,1])
    else: 
        ax[idx].set_ylim([min(ys)/2,1])
    
    ax[idx].set_yscale('log')
    if log == True:
        ax[idx].set_xscale('log')
        ax[idx].set_xlim([1/2,max(xs*2)])
    if idx % num_rows == 0:
        ax[idx].set_ylabel("Number of participants")
    if idx in np.arange(num_cols*num_rows - num_cols, num_cols*num_rows + 1, 1):
        ax[idx].set_xlabel("Number of contacts")
    ax[idx].legend()
    if idx == 0:
        ax[idx].set_title(f'{0}-{buckets[idx]-1}')
    elif idx == len(buckets):
        ax[idx].set_title(f'{buckets[idx-1]}+')
    else: 
        ax[idx].set_title(f'{buckets[idx-1]}-{buckets[idx]-1}')

      
def log_bins(x, num_bins=5):
    """
    Returns log bins of contacts, A^m
    Input: Contacts -> np array, num_bins -> int
    Output: Geometric center of bins -> ndarray, values in bins -> ndarray
    """
    # count_zeros = np.sum(x[x==0])
    count_zeros = len([a for a in x if a==0])
    x = np.sort([a for a in x if a > 0])
    max1, min1 = np.log(np.ceil(max(x))), np.log(np.floor(min(x)))
    x = np.log(x)
    t, freq, ends = np.zeros(num_bins), np.zeros(num_bins), np.zeros((2,num_bins))
    step = (max1 - min1)/num_bins
    for val in x:
        for k in range(num_bins):
            if k*step + min1 <= val and val < (k+1)*step + min1:
                freq[k] += 1
            t[k] = (k+1)*step - (.5*step) + min1
            ends[0,k] = k*step + min1
            ends[1,k] = (k+1)*step + min1
    freq[0] += count_zeros
    ends = np.exp(ends)
    widths = ends[1] - ends[0]
    freq = freq/widths/(len(x)+count_zeros)
    # freq = 1/np.sqrt(freq)*freq
    midpoints = np.exp(t)
    return midpoints, freq
    
# def calc_error(egos, network, distance_matrix, extra_mass_penalty, num_per_bucket):
    
#     # Begin parallelisation
#     # Number of threads
#     n = len(num_per_bucket)
#     # check that network is big enough 
#     for i in range(n):
#         if num_per_bucket[i] > network['partitions'][i]:
#             print("The network is smaller than the data.")
#             return None
    
#     # Data for each thread
#     data = [[] for _ in range(n)]
#     for ego in egos:
#         data[ego['age']].append(np.array(ego['contacts'], dtype=np.float64))
        
#     # network egos for each thread, big nasty line but it just samples randomly from the network to match age distribution of data
#     indices = [random.sample(range(0, network['partitions'][i]), num_per_bucket[i]) if i == 0 else random.sample(range(network['partitions'][i-1], network['partitions'][i]), num_per_bucket[i]) for i in range(n)]
#     network_data = [[np.array(network['frequency_distribution'][idx], dtype=np.float64) for idx in age_class] for age_class in indices]
    
#     # run calculation of the W matrix then pool and collect results
#     with Pool(n) as pool:
#         combined_args = zip(data, network_data, [distance_matrix for _ in range(n)], [extra_mass_penalty for _ in range(n)])

#         # map() preserves the order of results
#         W = pool.map(calc_error_bucket, combined_args)
        
#     # run matching problem in parallel and then pool and collect the results
#     with Pool(n) as pool:
#         total_errors = pool.map(solve_matching_problem, W)
        
#     # calculate total error per person
#     errors_per_person = np.sum(total_errors) / np.sum(num_per_bucket)
#     #normalise the error to be errors per person
#     error = np.divide(total_errors, num_per_bucket)
    
#     return error, errors_per_person

# def calc_error_bucket(arguments):
    
#     # Calculating the matrix of Wasserstein distances for each pair in an age group
#     data, network_data, distance_matrix, extra_mass_penalty = arguments
#     W_a = np.zeros((len(data), len(data)))
#     for i, ego in enumerate(data):
#         for j, ego_node in enumerate(network_data):
#             # extra mass penalty = A/(mean_mass + 1)
#             penalty = extra_mass_penalty / ((np.sum(ego) + np.sum(ego_node))/2 + 1)
#             W_a[i,j] = pyemd.emd(ego, ego_node, distance_matrix, extra_mass_penalty=penalty)
#     return W_a

# def solve_matching_problem(W_a):
    
#     # find the optimal matching of the W matrix
#     row_ind, col_ind = sc.optimize.linear_sum_assignment(W_a)
#     return W_a[row_ind, col_ind].sum()
