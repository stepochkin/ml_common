from stepin.testing.file_test import FileTest

import numpy as np
from numpy.testing.utils import assert_array_equal

from stepin.file_utils import str2file
from stepin.zstd_utils import read_str_csv_with_optional_cols


class ServiceTest(FileTest):
    def test_read_str_csv_with_optional_cols(self):
        path = self.tpath('data.csv')
        data = '''\
c1,c3
aa,11
bb,22
cc,33'''
        str2file(data, path, encoding='utf-8')
        columns = ['c1', 'c2', 'c3', 'c4']
        df = read_str_csv_with_optional_cols(path, columns)
        self.assertSetEqual(set(df.columns), set(columns))
        assert_array_equal(df[columns].values, np.array([
            ['aa', '', '11', ''],
            ['bb', '', '22', ''],
            ['cc', '', '33', ''],
        ]))
