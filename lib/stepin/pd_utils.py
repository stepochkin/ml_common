import numpy as np
import pandas as pd


def df_chunk_sized_iter(df, size):
    df_len = len(df)
    count = 0
    while count < df_len:
        start = count
        count += size
        yield df.iloc[start: count]


def pd_unique_concat_pair(df1, df2, col_name):
    df1_ucol = df1[col_name]
    df2_ucol = df2[col_name]
    add_values = np.setdiff1d(df2_ucol, df1_ucol)
    return pd.concat([df1, df2[df2_ucol.isin(add_values)]])


def pd_unique_concat(dfs, col_name, full_set=None):
    df = pd.concat(dfs)
    df.drop_duplicates(subset=col_name, inplace=True)
    if (len(dfs) > 1) and (full_set is not None):
        df = df.join(full_set, on=col_name, how='inner')
        # df = df[np.isin(df[col_name], full_set)]
    return df


def read_csv(data_path, **csv_params):
    return pd.read_csv(data_path, **csv_params)


def read_col_names(data_path):
    return pd.read_csv(data_path, nrows=0).columns


def read_str_csv(data_path, **csv_params):
    col_names = csv_params.get('usecols')
    if col_names is None:
        col_names = read_col_names(data_path)
    if 'encoding' not in csv_params:
        csv_params['encoding'] = 'utf-8'
    return read_csv(
        data_path,
        dtype={cname: str for cname in col_names},
        na_filter=False,
        **csv_params
    )
