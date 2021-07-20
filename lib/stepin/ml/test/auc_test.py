import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from stepin.ml.auc import calc_gauc, calc_gauc_fast, calc_auc_fast


class AucTest(unittest.TestCase):
    def test_gauc(self):
        actual = calc_gauc(
            np.array(['a', 'b', 'a', 'c', 'b', 'a', 'c', 'b', 'a']),
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
            np.array([0.9, 0.1, 0.7, 0.2, 0.8, 0.3, 0.5, 0.6, 0.8]),
        )
        expected = ((2 * 0.75) + (2 * 1.0)) / 4
        self.assertEqual(expected, actual)

    def test_gauc_fast(self):
        size = 10 ** 4
        y_score = np.random.random(size).astype(np.float32)
        y_true = np.random.choice([1.0, 0.0], size, p=[0.1, 0.9]).astype(np.float32)
        group_labels = np.random.random_integers(0, high=int(size * 0.15), size=size)
        gauc_fast = calc_gauc_fast(group_labels, y_true, y_score)
        # print('calc_gauc_fast', gauc_fast)
        gauc = calc_gauc(group_labels, y_true, y_score)
        # print('calc_gauc', gauc)
        self.assertAlmostEqual(gauc_fast, gauc, places=4)

    def test_auc_fast(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        y_score = np.array([0.9, 0.1, 0.7, 0.2, 0.8, 0.3, 0.5, 0.6, 0.8], dtype=np.float32)
        expected = roc_auc_score(y_true, y_score)
        actual = calc_auc_fast(y_true, y_score)
        self.assertAlmostEqual(expected, actual, places=7)
