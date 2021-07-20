import csv
import io

import pandas as pd
import zstandard as zstd

from stepin.log_utils import safe_close


class ChunkedStreamWrapper(object):
    def __init__(self, path, **df_args):
        self.fh = open(path, 'rb')
        dctx = zstd.ZstdDecompressor()
        self.reader = dctx.stream_reader(self.fh)
        self.df = pd.read_csv(self.reader, **df_args)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.df)

    def close(self):
        if self.fh is None:
            return
        self.df = None
        safe_close(self.reader)
        self.reader = None
        safe_close(self.fh)
        self.fh = None


def read_zstd_csv(path, **df_args):
    if 'chunksize' in df_args:
        return ChunkedStreamWrapper(path, **df_args)
    with open(path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            return pd.read_csv(reader, **df_args)


def read_csv(data_path, **csv_params):
    if isinstance(data_path, str) and data_path.endswith('.zst'):
        return read_zstd_csv(data_path, **csv_params)
    else:
        return pd.read_csv(data_path, **csv_params)


def read_col_names(data_path, csv_params=None):
    if csv_params is None:
        csv_params = dict(nrows=0)
    else:
        csv_params = csv_params.copy()
        csv_params.pop('chunksize', None)
        csv_params.pop('nrows', None)
        csv_params['nrows'] = 0
    if data_path.endswith('.zst'):
        return read_zstd_csv(data_path, **csv_params).columns
    else:
        return pd.read_csv(data_path, **csv_params).columns


def read_str_csv(data_path, **csv_params):
    if 'encoding' not in csv_params:
        csv_params['encoding'] = 'utf-8'
    col_names = csv_params.get('usecols')
    if col_names is None:
        col_names = read_col_names(data_path, csv_params)
    return read_csv(
        data_path,
        dtype={cname: str for cname in col_names},
        na_filter=False,
        **csv_params
    )


def read_str_csv_with_optional_cols(data_path, usecols, **csv_params):
    col_names = set(read_col_names(data_path))
    usecols = set(usecols)
    df = read_str_csv(data_path, usecols=list(col_names & usecols), **csv_params)
    for cname in (usecols - col_names):
        df[cname] = ''
    return df


class TextWrapper(object):
    def __init__(self, writer):
        self.writer = writer

    def write(self, s):
        self.writer.write(s.encode())

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    # for pandas
    def __iter__(self):
        raise NotImplementedError()


def write_zstd_csv(path, df, **df_args):
    with open(path, 'wb') as fh:
        cctx = zstd.ZstdCompressor()
        with cctx.stream_writer(fh) as writer:
            df.to_csv(TextWrapper(writer), **df_args)


def zstd_csv_via_bytes(path, **df_args):
    with open(path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(fh.read())
        return pd.read_csv(io.BytesIO(decompressed), **df_args)


class ZstdWriter(object):
    def __init__(self, path, mode='wb', file_args=None):
        if file_args is None:
            file_args = {}
        self.fh = None
        self.fh = open(path, mode, **file_args)
        cctx = zstd.ZstdCompressor()
        self.cobj = cctx.compressobj()

    def write(self, s):
        self.fh.write(self.cobj.compress(s.encode()))

    def flush(self):
        self.fh.write(self.cobj.flush(zstd.COMPRESSOBJ_FLUSH_BLOCK))

    def close(self):
        if self.fh is not None:
            self.fh.write(self.cobj.flush())
            self.fh.close()
            self.fh = None

    def __del__(self):
        self.close()


class ZstdCsvWriter(object):
    def __init__(self, path, append=False):
        # print('ZstdCsvWriter', path)
        self.writer = ZstdWriter(path, mode='ab' if append else 'wb')
        self.csv_writer = csv.writer(self.writer, lineterminator='\n')

    def writerow(self, row):
        self.csv_writer.writerow(row)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
