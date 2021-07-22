import unittest

import numpy as np
import scipy.sparse as sp

from stepin.ml.recs_metrics import calc_batch_precision_recall_ndcg


class RecsMetricsTest(unittest.TestCase):
    def test_calc_batch_precision_recall_ndcg(self):
        precisions, recalls, ndcg = calc_batch_precision_recall_ndcg(
            np.array([
                [2, 1],
                [0, 3],
            ]),
            sp.csr_matrix([
                [1, 0, 1, 1],
                [0, 1, 0, 0],
            ], dtype=np.uint8)
        )
        # print(precisions)
        # print(recalls)
        self.assertEqual(precisions[0], 0.5)
        self.assertEqual(precisions[1], 0.25)
        self.assertEqual(recalls[0], 1 / 6)
        self.assertEqual(recalls[1], 1 / 6)
