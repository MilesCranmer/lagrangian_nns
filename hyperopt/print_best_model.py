"""Start a hyperoptimization from a single node"""
import sys
import numpy as np
import pickle as pkl
import hyperopt
from hyperopt import hp, fmin, tpe, Trials


#Change the following code to your file
################################################################################
# TODO: Declare a folder to hold all trials objects
TRIALS_FOLDER = sys.argv[1]
################################################################################

def merge_trials(trials1, trials2_slice):
    """Merge two hyperopt trials objects

    :trials1: The primary trials object
    :trials2_slice: A slice of the trials object to be merged,
        obtained with, e.g., trials2.trials[:10]
    :returns: The merged trials object

    """
    max_tid = 0
    if len(trials1.trials) > 0:
        max_tid = max([trial['tid'] for trial in trials1.trials])

    for trial in trials2_slice:
        tid = trial['tid'] + max_tid + 1
        hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
        hyperopt_trial[0] = trial

        if hyperopt_trial[0]['result']['status'] != 'ok':
            hyperopt_trial[0]['result']['loss'] = float('inf')

        hyperopt_trial[0]['tid'] = tid
        hyperopt_trial[0]['misc']['tid'] = tid
        for key in hyperopt_trial[0]['misc']['idxs'].keys():
            hyperopt_trial[0]['misc']['idxs'][key] = [tid]
        trials1.insert_trial_docs(hyperopt_trial) 
        trials1.refresh()
    return trials1

np.random.seed()

# Load up all runs:
import glob
path = TRIALS_FOLDER + '/*.pkl'
files = 0
for fname in glob.glob(path):

    trials_obj = pkl.load(open(fname, 'rb'))
    n_trials = trials_obj['n']
    trials_obj = trials_obj['trials']
    if files == 0: 
        trials = trials_obj
    else:
        trials = merge_trials(trials, trials_obj.trials[-n_trials:])
    files += 1


#print(files, 'trials merged')


all_trials = []

for trial in trials:
    loss = trial['result']['loss']
    all_trials.append([trial['result']['loss'], trial['misc']['vals']])

best_trials = sorted(all_trials, key=lambda l: l[0])

for t in reversed(best_trials):
    print(t[0], t[1])


cream = best_trials[:100]
for key in cream[0][1].keys():
    print(key, np.median([t[1][key][0] for t in cream]))


