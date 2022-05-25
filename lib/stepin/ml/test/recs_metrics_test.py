import unittest

import numpy as np
import scipy.sparse as sp

from stepin.ml.recs_metrics import calc_batch_precision_recall_ndcg, calc_intra_list_similarity, calc_personalization, \
    calc_intra_list_similarity_sp, calc_intra_list_similarity_sp_mt


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

    def test_intra_list_similarity(self):
        features = sp.csr_matrix([
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
        ], dtype=np.uint8)
        actual = calc_intra_list_similarity(
            np.array([
                [2, 4, 5],
                [3, 0, 5],
                [1, 4, 3],
            ]),
            features
        )

        expected = (
            (0.0 + 1.0 / (np.sqrt(2) * np.sqrt(2)) + 1.0 / (1.0 * np.sqrt(2))) / 3.0 +
            (1.0 / (1.0 * np.sqrt(3)) + 0.0 + 2.0 / (np.sqrt(3) * np.sqrt(2))) / 3.0 +
            (0.0 + 0.0 + 0.0) / 3.0
        ) / 3.0
        self.assertEqual(actual, expected)

    def test_intra_list_similarity_sp(self):
        features = sp.csr_matrix([
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
        ], dtype=np.uint8)
        actual = calc_intra_list_similarity_sp(
            np.array([
                [2, 4, 5],
                [3, 0, 5],
                [1, 4, 3],
            ]),
            features
        )

        expected = (
            (0.0 + 1.0 / (np.sqrt(2) * np.sqrt(2)) + 1.0 / (1.0 * np.sqrt(2))) / 3.0 +
            (1.0 / (1.0 * np.sqrt(3)) + 0.0 + 2.0 / (np.sqrt(3) * np.sqrt(2))) / 3.0 +
            (0.0 + 0.0 + 0.0) / 3.0
        ) / 3.0
        self.assertEqual(actual, expected)

    def test_intra_list_similarity_sp_mt(self):
        features = sp.csr_matrix([
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
        ], dtype=np.uint8)
        actual = calc_intra_list_similarity_sp_mt(
            np.array([
                [2, 4, 5],
                [3, 0, 5],
                [1, 4, 3],
            ]),
            features,
            proc_count=3
        )

        expected = (
            (0.0 + 1.0 / (np.sqrt(2) * np.sqrt(2)) + 1.0 / (1.0 * np.sqrt(2))) / 3.0 +
            (1.0 / (1.0 * np.sqrt(3)) + 0.0 + 2.0 / (np.sqrt(3) * np.sqrt(2))) / 3.0 +
            (0.0 + 0.0 + 0.0) / 3.0
        ) / 3.0
        self.assertEqual(actual, expected)

    def test_calc_personalization(self):
        recs = np.array([
            [0, 2, 4],
            [0, 3, 5],
            [5, 6, 7],
            [0, 4, 8],
        ], dtype=np.int32)
        actual = calc_personalization(recs)
        expected = (1.0 / 3.0 + 0.0 + 2.0 / 3.0 + 1.0 / 3.0 + 1.0 / 3.0 + 0.0) / 6.0
        self.assertAlmostEqual(actual, expected)
