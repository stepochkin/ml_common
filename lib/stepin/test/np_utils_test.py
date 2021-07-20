# coding=utf-8

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from stepin.np_utils import largest_indices, NumpyBuilder


class NpUtilsTest(unittest.TestCase):
    @staticmethod
    def test_largest_indices():
        actual = largest_indices(np.array([5, 4, 8, 2, 4, 7, 3, 2]), 3)
        assert_array_equal(np.array([2, 5, 0]), actual)


def _nb_check(nb, data):
    assert_array_equal(np.array(data), nb.array())


class NumpyBuilderTest(unittest.TestCase):
    @staticmethod
    def test_add():
        nb = NumpyBuilder(np.int32, 4)
        expected = []
        for i in range(14):
            nb.add(i)
            expected.append(i)
            _nb_check(nb, expected)
        nb.add_many(np.arange(14, 18))
        expected.extend(np.arange(14, 18))
        _nb_check(nb, expected)
        nb.add_many(np.arange(18, 20))
        expected.extend(np.arange(18, 20))
        _nb_check(nb, expected)
        for i in range(20, 33):
            nb.add_many(np.array([i]))
            expected.append(i)
            _nb_check(nb, expected)
        nb.add_many(np.array([33, 34]))
        expected.extend([33, 34])
        _nb_check(nb, expected)
        a = [35, 36, 37, 38]
        nb.add_many(np.array(a))
        expected.extend(a)
        _nb_check(nb, expected)
        a = [39, 40, 41, 42, 43]
        nb.add_many(np.array(a))
        expected.extend(a)
        _nb_check(nb, expected)
        a = [44, 45, 46, 47, 48, 49]
        nb.add_many(np.array(a))
        expected.extend(a)
        _nb_check(nb, expected)
