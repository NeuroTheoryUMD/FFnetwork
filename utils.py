"""Utility functions for fitting models"""

from __future__ import division

import numpy as np


def get_indices(num_folds, rng_seed, trial_ids):

    """
    Get training/xv indices
    
    Args:
        num_folds (scalar):
        rng_seed (scalar):
        trial_ids (numpy array):
        
    Returns:
        index_reps (list of numpy arrays): indices evenly split into num_folds
        trials_in_fold (list of numpy arrays): trial_ids included in each fold
        
    """

    np.random.seed(rng_seed)

    # pull out relevant info
    trials = np.unique(trial_ids)
    num_trials = len(trials)
    trials_per_fold = np.floor(num_trials / num_folds).astype('int')
    index_reps = [0 for _ in range(num_folds)]
    trial_indxs = np.random.permutation(num_trials)

    # distribute leftover trials into initial folds
    num_leftover_trials = num_trials - num_folds * trials_per_fold
    trials_in_fold = [0 for _ in range(num_folds)]
    for i in range(num_folds):
        if i < num_leftover_trials:
            # add one to trials_per_fold to take up slack in leftover trials
            trials_in_fold[i] = trial_indxs[i * (trials_per_fold + 1) +
                                            range(trials_per_fold + 1)]
        else:
            trials_in_fold[i] = trial_indxs[
                num_leftover_trials * (trials_per_fold + 1) +
                (i - num_leftover_trials) * trials_per_fold +
                range(trials_per_fold)]

    # populate index_reps list
    for i in range(num_folds):
        index_list = np.array([], dtype='int')
        for j in range(trials_in_fold[i].size):
            temp_trial_num = trials_in_fold[i][j]
            index_list = np.append(
                index_list,
                np.where(trial_ids == temp_trial_num)[1])
        index_reps[i] = index_list

    return index_reps, trials_in_fold
