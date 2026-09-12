# import sys
# import os
# sys.path.append(os.path.abspath('..'))
import nd_python_avon as nd_p
import numpy as np
import glob
import os
import re
import sys
import json
import sklearn.mixture
import math

n = 100_000
num_networks = 30

## Growth-rate sweeps for reconnect, the one survey with no _gr runs on disk.
##
## These are the only runs that can give the early growth rate r.  dur_gillesp_sc, which
## writes the _age_dur/_sc files, keeps no clock at all -- it samples which event happens
## next, never when -- and dur_gillesp, which writes _fin, advances a clock but reads it
## once per run, at the peak.  dur_gillesp_gr is instrumented for r: it records prevalence
## at the moment generation 3 first appears and again one day later, and stores the ratio.
## Everything else here (the GMM fits, num_dur, props, iterations, num_infec) matches
## r_durdist_sims.py and r_sims_sbm.py exactly, so the tau -> R0 calibration carries over.
##
## taus: 100 points evenly spaced in tau from 0 up to R0 = 5, per model, which is the
## same shape of grid the poly and comixb _gr sweeps use -- np.linspace(0, tau_max, 100).
## tau_max comes from inverting the measured tau -> R0 curve of reconnect's _fin runs
## already on disk, fitting R0 - 1 = c + a log1p(b tau) as the tau -> R0 band section of
## paper_figs_no_lockdown1.ipynb does: R0 = 5 lands at tau = 4.00 for the duration model
## and tau = 0.297 for the SBM.  A tau means something different in each, hence one grid
## each; in R0 the two grids come out at a mean step of 0.04 over R0 = 1-5, against
## 0.008-0.059 for the four sweeps poly and comixb already have.
TAUS = {
    'dur': np.linspace(0, 4.0, 100),
    'sbm': np.linspace(0, 0.3, 100),
}

# model tag -> filename suffix.  'dur' has no tag, as everywhere else:
# duration+ages/seir_sims/{data}_{k}_gr.json against {data}_{k}_sbm_gr.json
SUFFIXES = {'dur': '', 'sbm': '_sbm'}

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

per_partition = [a if i == 0 else a-partitions[i-1] for i, a in enumerate(partitions)]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
duration_labels = ['0-1 hour', '1-4 hours', '4+ hours']
datas = ['reconnect']

# both by default; `python r_gr_sims.py dur` or `python r_gr_sims.py sbm` runs one, which
# is how to split them across two cluster jobs
models = sys.argv[1:] or ['dur', 'sbm']
assert all(m in TAUS for m in models), f'models must be from {sorted(TAUS)}'


def claim_index(data, suffix):
    """Reserve the next unused replicate index, and create the file to hold it.

    New runs continue past whatever is already on disk instead of overwriting it.
    This loop used to start from 0 every time, so resubmitting the job destroyed the
    first `num_networks` results of the previous batch.

    The empty placeholder claims the index immediately, so that a resubmitted job --
    or a second job on the same pair -- cannot pick the same one while this network
    is still simulating.  Aggregators skip files they cannot parse, so a placeholder
    left behind by a job that died is ignored rather than counted.

    The two models index independently: the glob catches both {data}_{k}_gr.json and
    {data}_{k}_sbm_gr.json, and the pattern then keeps only this model's.
    """
    pattern = re.compile(rf'^{data}_(\d+){suffix}_gr\.json$')
    used = [-1]
    for path in glob.glob(f'duration+ages/seir_sims/{data}_*_gr.json'):
        match = pattern.match(os.path.basename(path))
        if match:
            used.append(int(match.group(1)))
    k = max(used) + 1
    open(f'duration+ages/seir_sims/{data}_{k}{suffix}_gr.json', 'w').close()
    return k


def gmm_samples(egos, optimal_num_components, data):
    """One replicate's degree sample: refit the GMM per age bucket, then draw from it.

    Refitting rather than reusing a fit is what makes replicate networks differ, so this
    is inside the per-network loop in the sweeps this mirrors.  Fits are on log(k + 1)
    and inverted on the way out, with negatives clipped to zero.
    """
    samples = []
    for l, _ in enumerate(partitions):
        classifier = sklearn.mixture.GaussianMixture(
            n_components=optimal_num_components[data][l], covariance_type='full')
        egos_age = [a for a in egos if a['age'] == l]
        ## use log(k+1) instead of k to fit
        X = [[math.log(b+1) for b in a['contacts']] for a in egos_age]
        classifier.fit(X)
        ## sample same number of people as the data
        samples_tmp, _ = classifier.sample(per_partition[l])
        for sample in samples_tmp:
            samples.append([int(np.round(np.exp(b)-1)) if int(np.round(np.exp(b)-1)) >= 0 else 0
                            for b in sample])
    return samples


for model in models:
    suffix, taus = SUFFIXES[model], TAUS[model]
    print(f'=== {model}: {len(taus)} taus, {taus[0]:.4g} to {taus[-1]:.4g}, '
          f'{num_networks} networks per dataset', flush=True)

    for i, data in enumerate(datas):
        if model == 'dur':
            with open(f'duration+ages/data/gmm_opt_comp/optimal_components_{data}_log_smalldur.json', 'r') as f:
                optimal_num_components = json.load(f)
            ##################### read fits ####################################
            with open(f'input_data/egos/{data}_dur_small.json', 'r') as f:
                egos = json.load(f)
            props = np.genfromtxt(f'input_data/durations/{data}.csv', delimiter=',')
        else:
            cm = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')

        for _ in range(num_networks):
            k = claim_index(data, suffix)
            print(f'network {k} for data {data} ({model})', flush=True)
            if model == 'dur':
                samples = gmm_samples(egos, optimal_num_components, data)
                res = nd_p.gmm_dur_gillesp_gr(samples, partitions=partitions, num_dur=3,
                                              taus=taus, iterations=48,
                                              props=props.tolist(), num_infec=1)
            else:
                res = nd_p.sbm_gillesp_gr(contact_matrix=cm, partitions=partitions,
                                          taus=taus, iterations=48, num_infec=1)
            with open(f'duration+ages/seir_sims/{data}_{k}{suffix}_gr.json', 'w') as f:
                json.dump(res, f)
