# coding=utf-8

from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

from stepin.log_utils import safe_close
from stepin.np_utils import build_sets_array, build_1item_sets_array


def calc_ndcg(pred_indices, true, topk=None, calc_mean=True):
    if topk is None:
        topk = pred_indices.shape[1]
        pred_indices = pred_indices[:, :topk]
    discount = 1 / (np.log2(np.arange(2, topk + 2)))
    discount = discount.reshape([1, -1])
    res = []
    for i in range(topk):
        pred1 = build_1item_sets_array(pred_indices[:, i], true.shape[1])
        pred1 = true.multiply(pred1).sum(axis=1)
        res.append(np.asarray(pred1))
    res = np.column_stack(res)
    dcg = np.multiply(res, discount).cumsum(axis=1)
    res[:, ::-1].sort()
    idcg = np.multiply(res, discount).cumsum(axis=1)
    ndcg = np.zeros_like(idcg)
    np.divide(dcg, idcg, out=ndcg, where=idcg > 0.0)
    if calc_mean:
        ndcg = ndcg.mean(axis=0)
    return ndcg


def calc_batch_precision_recall_ndcg(pred_indices, true, topk=None):
    if topk is None:
        topk = pred_indices.shape[1]
        pred_indices = pred_indices[:, :topk]
    tp = []
    for i in range(topk):
        pred1 = build_1item_sets_array(pred_indices[:, i], true.shape[1])
        pred1 = true.multiply(pred1).sum(axis=1)
        tp.append(np.asarray(pred1))
    tp = np.column_stack(tp)
    tps = tp.cumsum(axis=1)
    precisions = tps / np.arange(1, topk + 1).reshape([1, -1])
    precisions = precisions.mean(axis=0)
    recalls = tps / true.getnnz(axis=1).reshape([-1, 1])
    recalls = recalls.mean(axis=0)
    discount = 1 / (np.log2(np.arange(2, topk + 2)))
    discount = discount.reshape([1, -1])
    dcg = np.multiply(tp, discount).cumsum(axis=1)
    tp[:, ::-1].sort()
    idcg = np.multiply(tp, discount).cumsum(axis=1)
    ndcg = np.zeros_like(idcg)
    np.divide(dcg, idcg, out=ndcg, where=idcg > 0.0)
    ndcg = ndcg.mean(axis=0)
    return precisions, recalls, ndcg


def select_first_cols(arr, numbers):
    arr = arr[numbers[:, None] > np.arange(arr.shape[1])]
    if len(arr.shape) > 1:
        arr = np.array(arr).reshape(-1)
    return arr


def first_cols_matrix(items, item_lens, matrix_width):
    m = sp.lil_matrix((len(item_lens), matrix_width), dtype=items.dtype)
    # m = np.zeros((len(item_lens), matrix_width), dtype=items.dtype)
    m[item_lens[:, None] > np.arange(matrix_width)] = items
    return m.tocsr()


def calc_batch_tp_fn_fast(
    test_indices, test_indices_lens, pred_indices, pred_indices_lens, item_count,
    thread_count=1
):
    """
    Calculates True-Positive and False-Negative counts for predictions.

    Parameters
    ----------
    test_indices : numpy array (N,) - items from test set
        where N is a number of items in test set and dtype is any numpy integer type.
    test_indices_lens : numpy array (K,) - number of items in test_indices for each user
        where K is a number of users in test set and dtype is any numpy integer type.
    pred_indices : numpy array (N,) - reverse sorted predictions by importance
        where N is a number of items not in train set and dtype is any numpy integer type.
    pred_indices_lens : numpy array (K,) - number of items in pred_indices for each user
        where K is a number of users in test set and dtype is any numpy integer type.
    item_count : total number of unique items.
    thread_count : number of execution threads

    Returns
    -------
    numpy array (recall_points count, 2) dtype = int
        Each row contains recall_point: [TruePositive count, FalseNegative count].

    Examples
    --------
    calc_batch_tp_fn_fast(
        np.array([
            2, 7, 
            4, 6, 8
        ]), 
        np.array([2, 3]), 
        np.array([
            5, 7, 4, 1,
            3, 6, 4, 5, 1,
        ]), 
        np.array([4, 5]),
        10
    )
    """

    user_count = len(test_indices_lens)
    test_sets = build_sets_array(test_indices, test_indices_lens, user_count, item_count)
    test_lens_sum = len(test_indices)
    pred_matrix = first_cols_matrix(pred_indices, pred_indices_lens, item_count)
    max_x = np.max(pred_indices_lens)
    if thread_count < 2:
        tps = [
            _calc_single_tp(x, user_count, item_count, pred_indices_lens, pred_matrix, test_sets)
            for x in range(1, max_x + 1)
        ]
    else:
        global _pred_indices_lens, _pred_matrix, _test_sets
        _pred_indices_lens = pred_indices_lens
        _pred_matrix = pred_matrix
        _test_sets = test_sets
        pool = Pool(processes=thread_count)
        try:
            tps = pool.map(
                _calc_single_tp_sargs,
                (
                    (x, user_count, item_count, _pred_indices_lens, _pred_matrix, _test_sets)
                    for x in range(1, max_x + 1)
                )
            )
        finally:
            safe_close(pool)
    # tps = np.array(tps, dtype=np.uint64)
    return [np.column_stack([tps_k, test_indices_lens - tps_k]) for tps_k in tps]

def calc_precision_recall(
    test_indices, test_indices_lens, pred_indices, pred_indices_lens, item_count,
    thread_count=1
):
    all_tp_fp = calc_batch_tp_fn_fast(
        test_indices, test_indices_lens, pred_indices, pred_indices_lens, item_count,
        thread_count=thread_count
    )
    recalls = []
    grecalls = []
    precs = []
    gprecs = []
    for i in range(len(all_tp_fp)):
        tp_fp = all_tp_fp[i]
        k = i + 1
        test_sizes = tp_fp.sum(axis=1)
        tp = tp_fp[:, 0]
        recalls.append((tp / test_sizes).mean())
        grecalls.append(tp.sum() / test_sizes.sum())
        precs.append((tp / k).mean())
        gprecs.append(tp.sum() / (k * tp.shape[0]))
    return recalls, grecalls, precs, gprecs

_pred_indices_lens = None
_pred_matrix = None
_test_sets = None


def _calc_single_tp(x, user_count, item_count, pred_indices_lens, pred_matrix, test_sets):
    select_lens = pred_indices_lens.copy()
    select_lens[select_lens > x] = x
    selected_items = select_first_cols(pred_matrix, select_lens)
    selected_sets = build_sets_array(
        selected_items, select_lens, user_count, item_count
    )
    intersect_sets = test_sets.multiply(selected_sets)
    tp = np.array(intersect_sets.sum(axis=1)).reshape(-1)
    return tp


def _calc_single_tp_sargs(args):
    return _calc_single_tp(*args)
