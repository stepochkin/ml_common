import numpy as np
from stepin.np_utils import build_1item_sets_array


def sample_rand_negs(user_positives, dtype=np.int32, allowed=None):
    item_count = user_positives.shape[1]
    sample_count = item_count if allowed is None else allowed.shape[1]
    neg_items = np.random.randint(0, sample_count, user_positives.shape[0], dtype=dtype)
    if allowed is not None:
        neg_items = allowed[np.arange(allowed.shape[0]), neg_items]
    # noinspection PyUnresolvedReferences
    diff_poss = np.nonzero(
        user_positives.multiply(build_1item_sets_array(neg_items, item_count)).sum(axis=1)
    )[0]
    while len(diff_poss) > 0:
        ni = np.random.randint(0, sample_count, len(diff_poss), dtype=dtype)
        if allowed is not None:
            ni = allowed[diff_poss, ni]
        neg_items[diff_poss] = ni
        diff_counts = user_positives[diff_poss].multiply(
            build_1item_sets_array(ni, item_count)
        ).sum(axis=1)
        # noinspection PyUnresolvedReferences
        diff_poss = diff_poss[np.nonzero(diff_counts)[0]]
    return neg_items
