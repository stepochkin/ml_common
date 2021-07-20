# coding=utf-8

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def read_npz(path, fname='data', is_sparse=False):
    with np.load(path) as f:
        if is_sparse:
            return csr_matrix(
                (f[fname + '_data'], f[fname + '_indices'], f[fname + '_indptr']),
                shape=f[fname + '_shape']
            )
        return f[fname]


def read_npzs(path, *fnames, sp_names=None, load_params=None):
    if load_params is None:
        load_params = {}
    with np.load(path, **load_params) as f:
        arrays = [f[fname] for fname in fnames]
        if sp_names is None:
            return arrays
        sp_arrays = [
            csr_matrix(
                (f[fname + '_data'], f[fname + '_indices'], f[fname + '_indptr']),
                shape=f[fname + '_shape']
            ) for fname in sp_names
        ]
        return arrays + sp_arrays


def write_npzs(path, sp_arrays=None, **arrays):
    if sp_arrays is not None:
        for fname, arr in sp_arrays.items():
            arrays[fname + '_data'] = arr.data
            arrays[fname + '_indices'] = arr.indices
            arrays[fname + '_indptr'] = arr.indptr
            arrays[fname + '_shape'] = arr.shape
    np.savez_compressed(path, **arrays)


def read_npz_cond(path, fname1, fname2):
    with np.load(path) as f:
        if fname1 in f.files:
            return f[fname1]
        return f[fname2]


def select_first_cols(arr, numbers):
    arr = arr[numbers[:, None] > np.arange(arr.shape[1])]
    if len(arr.shape) > 1:
        arr = np.array(arr).reshape(-1)
    return arr


def first_cols_matrix(items, item_lens, matrix_width):
    m = lil_matrix((len(item_lens), matrix_width), dtype=items.dtype)
    # m = np.zeros((len(item_lens), matrix_width), dtype=items.dtype)
    m[item_lens[:, None] > np.arange(matrix_width)] = items
    return m.tocsr()


def first_cols_csr(arr, lengths):
    fc = select_first_cols(arr, lengths)
    return csr_matrix(
        (np.ones(fc.shape, dtype=np.int32), fc, np.r_[0, lengths.cumsum()]),
        shape=arr.shape
    )


def calc_group_lengths(arr, return_values=False, return_val_len=False):
    d = np.diff(arr)
    # noinspection PyUnresolvedReferences
    n = np.nonzero(d)[0]
    if len(n) == 0:
        dn = np.array([len(arr)])
        if return_values:
            return dn, arr[0]
        if return_val_len:
            return dn, 1
        return dn
    dn = np.diff(n)
    dn = np.r_[n[0] + 1, dn, len(arr) - n[-1] - 1]
    if return_values:
        return dn, np.r_[arr[n], arr[-1]]
    if return_val_len:
        return dn, len(n) + 1
    return dn


def group_indices(arr):
    diff = arr[1:] != arr[:-1]
    return np.r_[True, diff].nonzero()[0]


def csr_group_indices(arr):
    diff = arr[1:] != arr[:-1]
    return np.r_[True, diff, True].nonzero()[0]


def group_values(arr):
    return arr[np.insert(np.diff(arr).astype(np.bool), 0, True)]


def vrange(starts, stops):
    """Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    stops = np.asarray(stops)
    lens = stops - starts  # Lengths of each range.
    return np.repeat(stops - lens.cumsum(), lens) + np.arange(lens.sum())


def multirange(lens):
    lens = np.asarray(lens)
    return np.arange(lens.sum()) - np.repeat(np.r_[0, lens[:-1].cumsum()], lens)


# def multirange(counts):
#     counts = np.asarray(counts)
#     # Remove the following line if counts is always strictly positive.
#     counts = counts[counts != 0]
#
#     counts1 = counts[:-1]
#     reset_index = np.cumsum(counts1)
#
#     incr = np.ones(counts.sum(), dtype=int)
#     incr[0] = 0
#     incr[reset_index] = 1 - counts1
#
#     incr.cumsum(out=incr)
#     return incr


def largest_indices(arr, n):
    indices = np.argpartition(arr, -n)[-n:]
    indices = indices[np.argsort(-arr[indices])]
    return indices


def matrix2array1d(m):
    return np.squeeze(np.array(m))


def group_unique(arr):
    return arr[np.insert(np.diff(arr).astype(np.bool), 0, True)]


def build_sets_array(values, lens, height, width):
    if height is None:
        height = len(lens)
    return csr_matrix(
        (
            np.ones(len(values), dtype=np.int8),
            values,
            np.r_[0, np.cumsum(lens)]
        ),
        shape=(height, width)
    )


def build_1item_sets_array(values, width, dtype=np.int8):
    return csr_matrix(
        (
            np.ones(len(values), dtype=dtype),
            values,
            np.arange(len(values) + 1)
        ),
        shape=(len(values), width)
    )


def csr_repeat(data, lens, width):
    data = np.split(data, np.cumsum(lens[:-1]))
    data = np.repeat(data, lens)
    lens = np.repeat(lens, lens)
    return build_sets_array(np.hstack(data), lens, None, width)


class NumpyBuilder(object):
    def __init__(self, dtype, chunk_size, col_count=None):
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.col_count = col_count
        self.chunk = self._new_chunk()
        self.chunk_pos = None
        self.chunks = None
        self.reset()

    def _new_chunk(self):
        return np.empty(
            self.chunk_size if self.col_count is None else [self.chunk_size, self.col_count],
            dtype=self.dtype
        )

    def reset(self):
        self.chunk_pos = 0
        self.chunks = []

    def add(self, row):
        if self.chunk_pos == self.chunk_size:
            self.chunks.append(self.chunk)
            self.chunk = self._new_chunk()
            self.chunk_pos = 0
        self.chunk[self.chunk_pos] = row
        self.chunk_pos += 1

    def add_many(self, rows):
        rshape = rows.shape[0]
        if self.chunk_pos + rshape - 1 >= self.chunk_size:
            size1 = self.chunk_size - self.chunk_pos
            self.chunk[self.chunk_pos:] = rows[:size1]
            self.chunks.append(self.chunk)
            self.chunk_pos = rshape - size1
            if self.chunk_pos > self.chunk_size:
                self.chunks.append(np.array(rows[size1:], dtype=self.dtype))
                self.chunk = self._new_chunk()
                self.chunk_pos = 0
            else:
                self.chunk = self._new_chunk()
                self.chunk[:self.chunk_pos] = rows[size1:]
        else:
            self.chunk[self.chunk_pos: self.chunk_pos + rshape] = rows
            self.chunk_pos += rshape

    def array(self):
        if len(self.chunks) == 0:
            return self.chunk[:self.chunk_pos]
        if self.col_count is None:
            return np.hstack(self.chunks + [self.chunk[:self.chunk_pos]])
        else:
            return np.vstack(self.chunks + [self.chunk[:self.chunk_pos]])
