# import sys
# import os
# sys.path.append(os.path.abspath('..'))
import nd_python_avon as nd_p
import numpy as np
import glob
import os
import re
import json

n = 100_000
num_networks= 20

## same sweep as the other *_r0_comparison.json runs, so a tau index means the
## same transmission rate in every dataset and the R0s can be compared directly
taus = np.linspace(0, 7, 200)
iterations = 48

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

bucket_labels = ['0-4', '5-11', '12-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
datas = ['reconnect']

## the SBM counterpart of r_r0_comparison.py: the same taus and iterations, but the
## network is built from the contact matrix rather than the fitted degree mixture,
## so no egos, GMM components or duration proportions are read.
##
## The suffix keeps these out of the GMM+duration pool.  Figure 4 looks for
## {data}_{k}_r0_comparison.json exactly, so it ignores these rather than averaging
## two different models into one curve; read them deliberately to compare the two.
SUFFIX = '_sbm'

## the other *_r0_comparison.json files hold only these; sbm_gillesp_sc also
## returns age_dur_sc, which is dropped to match them (and to keep the files small)
keys = ['sc', 'sc2', 'sc3', 'taus']


def claim_index(data):
    """Reserve the next unused replicate index, and create the file to hold it.

    New runs continue past whatever is already on disk instead of overwriting it.
    r_r0_comparison.py instead offsets by a fixed num_networks, which is why the
    indices already on disk come in blocks with gaps between them: a resubmission
    lands on the previous block unless the offset is edited by hand.

    The empty placeholder claims the index immediately, so that a resubmitted job --
    or a second job on the same dataset -- cannot pick the same one while this
    network is still simulating.  Aggregators skip files they cannot parse, so a
    placeholder left behind by a job that died is ignored rather than counted.
    """
    pattern = re.compile(rf'^{data}_(\d+){SUFFIX}_r0_comparison\.json$')
    used = [-1]
    for path in glob.glob(f'duration+ages/seir_sims/{data}_*{SUFFIX}_r0_comparison.json'):
        match = pattern.match(os.path.basename(path))
        if match:
            used.append(int(match.group(1)))
    k = max(used) + 1
    open(f'duration+ages/seir_sims/{data}_{k}{SUFFIX}_r0_comparison.json', 'w').close()
    return k


for i, data in enumerate(datas):
    ##################### read contact matrix ##########################
    cm = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')

    for _ in range(num_networks):
        k = claim_index(data)
        print(f'network {k} for data {data}', flush=True)
        res = nd_p.sbm_gillesp_sc(contact_matrix=cm, partitions=partitions, taus=taus, iterations=iterations, num_infec=1)
        with open(f'duration+ages/seir_sims/{data}_{k}{SUFFIX}_r0_comparison.json','w') as f:
            json.dump({key: res[key] for key in keys}, f)
