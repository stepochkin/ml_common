# distutils: language = c++

import ctypes
import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.vector cimport vector


cdef struct TFP:
    float tp
    float fp


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _gauc_group_c(const vector[TFP] &tps, int i, int gfi, float yts, float *yts_all, float *result):
    cdef size_t gi
    cdef float g_all_tp = yts
    cdef float g_all_fp = i - gfi - yts
    if g_all_tp <= 0.0 or g_all_fp <= 0.0:
        return
    cdef float tpr, fpr
    cdef float tpr_prev = 0.0
    cdef float fpr_prev = 0.0
    gauc = 0.0
    gi = 0
    while gi < tps.size():
        tpr = tps[gi].tp / g_all_tp
        fpr = tps[gi].fp / g_all_fp
        gauc += ((fpr - fpr_prev) * (tpr + tpr_prev)) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr
        inc(gi)
    gauc += ((1.0 - fpr_prev) * (1.0 + tpr_prev)) / 2.0
    gauc *= yts
    yts_all[0] += yts
    result[0] += gauc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float gauc_c(
    np.ndarray[np.int64_t, ndim=1] g,
    np.ndarray[np.float32_t, ndim=1] yt,
    np.ndarray[np.float32_t, ndim=1] ys
) except*:
    cdef np.int64_t g_cur, g_prev = g[0]
    cdef float ys_cur, ys_prev = ys[0]
    cdef vector[TFP] tps
    cdef float gauc, result = 0.0
    cdef int i, gfi = 0
    cdef float yts = 0.0, yts_all = 0.0
    for i in range(g.shape[0]):
        g_cur = g[i]
        if g_cur != g_prev:
            _gauc_group_c(tps, i, gfi, yts, &yts_all, &result)
            yts = 0.0
            gfi = i
            tps.resize(0)
            g_prev = g_cur
        ys_cur = ys[i]
        if i != gfi and ys_prev != ys_cur:
            tps.push_back(TFP(tp = yts, fp = i - gfi - yts))
            ys_prev = ys_cur
        yts += yt[i]
    _gauc_group_c(tps, g.shape[0], gfi, yts, &yts_all, &result)
    result /= yts_all
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def gauc(
    np.ndarray[np.int64_t, ndim=1] g,
    np.ndarray[np.float32_t, ndim=1] yt,
    np.ndarray[np.float32_t, ndim=1] ys
):
    return gauc_c(g, yt, ys)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float auc_c(
    np.ndarray[np.float32_t, ndim=1] yt,
    np.ndarray[np.float32_t, ndim=1] ys
) except*:
    cdef float ys_cur, ys_prev = ys[0]
    cdef vector[TFP] tps
    cdef int i
    cdef float yts = 0.0
    for i in range(yt.shape[0]):
        ys_cur = ys[i]
        if i != 0 and ys_prev != ys_cur:
            tps.push_back(TFP(tp = yts, fp = i - yts))
            ys_prev = ys_cur
        yts += yt[i]

    cdef float g_all_tp = yts
    cdef float g_all_fp = yt.shape[0] - yts
    if g_all_tp <= 0.0 or g_all_fp <= 0.0:
        raise Exception('All y_true have the same value')
    cdef float tpr, fpr
    cdef float tpr_prev = 0.0
    cdef float fpr_prev = 0.0
    cdef float auc = 0.0
    cdef size_t ui = 0
    while ui < tps.size():
        tpr = tps[ui].tp / g_all_tp
        fpr = tps[ui].fp / g_all_fp
        auc += ((fpr - fpr_prev) * (tpr + tpr_prev)) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr
        inc(ui)
    auc += ((1.0 - fpr_prev) * (1.0 + tpr_prev)) / 2.0
    return auc


@cython.boundscheck(False)
@cython.wraparound(False)
def auc(
    np.ndarray[np.float32_t, ndim=1] yt,
    np.ndarray[np.float32_t, ndim=1] ys
):
    return auc_c(yt, ys)
