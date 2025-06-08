from __future__ import division, print_function, absolute_import
import numpy as np
import warnings

def calculate_minp(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Calculates the mean Inverse Negative Penalty (mINP) metric.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int): maximum rank to consider. Not directly used for mINP calculation
                       but kept for consistency with other eval functions if needed.

    Returns:
        float: The mean Inverse Negative Penalty (mINP).
    """
    num_q, num_g = distmat.shape

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_inp = []
    num_valid_q = 0  # number of valid queries

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # get the matches for the kept gallery samples
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches

        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            # or all gallery matches are from the same camera as the query
            continue

        num_valid_q += 1
        num_rel = raw_cmc.sum() # |G_i|: total number of relevant gallery images for this query

        if num_rel == 0:
            # Should not happen if np.any(raw_cmc) is true, but handle defensively
            all_inp.append(0.0)
            continue

        # Find the rank of the hardest positive (last match)
        # np.where returns a tuple of arrays; we need the first array
        match_positions = np.where(raw_cmc == 1)[0]
        r_hard = match_positions[-1] + 1 # Rank is 1-based index

        # Calculate INP_i = |G_i| / R_hard_i
        inp_i = num_rel / r_hard
        all_inp.append(inp_i)

    if num_valid_q == 0:
        warnings.warn("No valid queries found for mINP calculation. Returning 0.")
        return 0.0

    mINP = np.mean(all_inp)
    return mINP
