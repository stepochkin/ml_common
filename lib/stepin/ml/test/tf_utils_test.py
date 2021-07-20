import unittest

import numpy as np
from datetime import datetime
from numpy.testing.utils import assert_array_equal
import tensorflow as tf

from stepin.ml.tf_utils import repeat_range


class TfUtilsTest(unittest.TestCase):
    @staticmethod
    def test_repeat_range():
        op = repeat_range([4, 8])
        sess = tf.Session()
        actual = sess.run(op)
        expected = np.repeat(np.arange(2), [4, 8])
        assert_array_equal(expected, actual)

    # @staticmethod
    # def test_time():
    #     op = repeat_range(np.repeat([100], 10000))
    #     sess = tf.Session()
    #     start = datetime.now()
    #     sess.run(op)
    #     print('')
    #     print(datetime.now() - start)
