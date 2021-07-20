#!/usr/bin/env python3

from datetime import datetime

import numpy as np
from nose.tools import assert_almost_equal
from sklearn.metrics import roc_auc_score

from stepin.ml.auc import calc_gauc, calc_gauc_fast, calc_auc_fast


def _main():
    size = 10**5
    y_score = np.random.random(size).astype(np.float32)
    y_true = np.random.choice([1.0, 0.0], size, p=[0.1, 0.9]).astype(np.float32)
    group_labels = np.random.random_integers(0, high=int(size * 0.15), size=size)
    start_dt = datetime.now()
    gauc_fast = calc_gauc_fast(group_labels, y_true, y_score)
    print((datetime.now() - start_dt).total_seconds())
    # print('calc_gauc_fast', gauc_fast)
    start_dt = datetime.now()
    gauc = calc_gauc(group_labels, y_true, y_score)
    print((datetime.now() - start_dt).total_seconds())
    # print('calc_gauc', gauc)
    assert_almost_equal(gauc_fast, gauc, places=4)

    start_dt = datetime.now()
    auc_fast = calc_auc_fast(y_true, y_score)
    print((datetime.now() - start_dt).total_seconds())
    # print('calc_gauc_fast', gauc_fast)
    start_dt = datetime.now()
    auc = roc_auc_score(y_true, y_score)
    print((datetime.now() - start_dt).total_seconds())
    # print('calc_gauc', gauc)
    assert_almost_equal(auc_fast, auc, places=4)


_main()
