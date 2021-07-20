# coding=utf-8

import codecs
import csv
import gzip

import numpy as np

from stepin.np_utils import calc_group_lengths


def memory_batcher(*x_, y_=None, batch_size=-1, n_epochs=1, always_tuple=False):
    """Split data to mini-batches.

    Parameters
    ----------
    x_ : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y_ : np.array or None, shape (n_samples,)
        Target vector relative to X.

    batch_size : int
        Size of batches.
        Use -1 for full-size batches
        
    n_epochs : int
        Epoch quantity

    always_tuple: bool

    Yields
    -------
    ret_x : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Same type as input
    ret_y : np.array or None, shape (batch_size,)
    """
    n_samples = x_[0].shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for epoch_i in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            upper_bound = min(i + batch_size, n_samples)
            if not always_tuple and (len(x_) == 1):
                ret_x = x_[0][i:upper_bound]
            else:
                ret_x = tuple(xx[i:upper_bound] for xx in x_)
            if y_ is None:
                yield ret_x
            else:
                if not always_tuple and (len(x_) == 1):
                    yield (ret_x, y_[i:i + batch_size])
                else:
                    yield ret_x + (y_[i:i + batch_size],)


def group_memory_batcher(groups, data, batch_size=-1, n_epochs=1):
    groups = calc_group_lengths(groups)
    n_samples = groups.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    prev_pos = 0
    for epoch_i in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            # print(i)
            upper_bound = min(i + batch_size, n_samples)
            batch_groups = groups[i: upper_bound]
            # print(batch_groups)
            batch_len = batch_groups.sum()
            ret_data = data[prev_pos: prev_pos + batch_len]
            yield ret_data
            prev_pos += batch_len


def _gzip_iter(path):
    with gzip.open(path) as f:
        for line in f:
            yield line.decode('utf-8')


def int_row_parser(row):
    return tuple(int(item) for item in row)


def text_batch_iter(path, batch_size, gzipped=False, row_parser=None, dtype=None, csv_params=None):
    if csv_params is None:
        csv_params = {}
    if gzipped:
        it = _gzip_iter(path)
    else:
        it = codecs.open(path, encoding='utf-8')
    it = csv.reader(it, **csv_params)
    try:
        row = next(it)
    except StopIteration:
        return
    if row_parser is not None:
        row = row_parser(row)
    batch = np.empty((batch_size, len(row)), dtype=dtype)
    batch[0] = row
    cur_size = 1
    if cur_size >= batch_size:
        yield batch
        cur_size = 0
    for row in it:
        if row_parser is not None:
            row = row_parser(row)
        batch[cur_size] = row
        cur_size += 1
        if cur_size >= batch_size:
            yield batch
            cur_size = 0
    if cur_size > 0:
        yield batch[:cur_size]
