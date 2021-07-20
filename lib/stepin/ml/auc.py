import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from stepin.ml.ml_ext import gauc as gauc_ext, auc as auc_ext


def calc_gauc(group_labels, y_true, y_score):
    df = pd.DataFrame(dict(g=group_labels, yt=y_true, ys=y_score))
    df = df.groupby('g')

    gauc = 0.0
    event_count = 0
    for n, g in df:
        # print(n)
        # print(g)
        gyt = g.yt
        # print('gyt', np.unique(gyt), len(np.unique(gyt)))
        if len(np.unique(gyt)) <= 1:
            continue
        group_auc = roc_auc_score(g.yt, g.ys)
        g_event_count = gyt.sum()  # (gyt == 1.0).sum()
        # print('gauc', group_auc, g_event_count)
        gauc += g_event_count * group_auc
        event_count += g_event_count
    # print('event_count', event_count)
    gauc /= event_count
    return gauc


def calc_gauc_fast(group_labels, y_true, y_score):
    df = pd.DataFrame(dict(g=group_labels, yt=y_true, ys=y_score))
    df.g = df.groupby('g').ngroup()
    df.sort_values(['g', 'ys'], ascending=[True, False], inplace=True)
    # print(df.head(20))
    # print(df.g.values.dtype)
    # print(df.yt.values.dtype)
    # print(df.ys.values.dtype)
    return gauc_ext(df.g.values, df.yt.values.astype(np.float32), df.ys.values)


def calc_auc_fast(y_true, y_score):
    # df = pd.DataFrame(dict(yt=y_true, ys=y_score))
    # df.sort_values(['ys'], ascending=[False], inplace=True)
    # y_true = df.yt.values
    # y_score = df.ys.values
    #
    sind = np.argsort(y_score)[::-1]
    y_true = y_true[sind]
    y_score = y_score[sind]
    return auc_ext(y_true.astype(np.float32), y_score.astype(np.float32))
