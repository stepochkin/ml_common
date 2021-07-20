#!/usr/bin/env python3

import argparse
import os
from collections import namedtuple

import numpy as np
import pandas as pd

from stepin.json_utils import read_json


def gen_w():
    p = 0.1**(1.0/49)
    w = []
    cp = 1.0
    for i in range(100):
        w.append(cp)
        cp *= p
    return np.array(w)


def add_param(v, s, default=None, width=8, formatter=None, ndigits=None):
    # print('add_param', v, default, width, ndigits)
    if v is None:
        sa = '??' if default is None else default
    else:
        if formatter is not None:
            v = formatter(v)
        else:
            if isinstance(v, float) and ndigits is not None:
                v = round(v, ndigits=ndigits)
        sa = '%s' % v
    sa = sa.ljust(width)
    return s + sa


def print_info(meta, info):
    s = ''
    for md in meta:
        f = md.get('format')
        if f is None:
            f = {}
        s = add_param(getattr(info, md['name']), s, **f)
    print(s)


def traverse_dict(d, names):
    v = d
    for n in names:
        v = v.get(n)
        if v is None:
            break
    return v


class DictValue(object):
    def __init__(self, names):
        self.names = names

    def get(self, info):
        v = traverse_dict(info, self.names)
        # if isinstance(v, float) and abs(v) > 1e-5:
        #     v = round(v, 5)
        return v


def get_meta_dict(md_names, name, formats, title=None):
    mdf = formats.get(name)
    if mdf is None:
        mdf = {}
    if len(name) > 7 and mdf.get('width') is None:
        mdf['width'] = len(name) + 2
    return dict(
        name=name if title is None else title,
        proc=DictValue(md_names + [name]).get,
        format=mdf,
    )

def add_meta_dict(meta, md_names, name, formats, title=None):
    m = get_meta_dict(md_names, name, formats, title=title)
    if m['name'] in meta:
        return
    meta[m['name']] = m

def fill_meta(meta, info_sample, formats=None, data_cols=None, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = set()
    if formats is None:
        formats = {}

    for names in (['params', 'model'],):
        d = traverse_dict(info_sample, names)
        if d is not None:
            for k in d:
                if k not in exclude_cols:
                    add_meta_dict(meta, names, k, formats)
    if data_cols is not None:
        for k in data_cols:
            add_meta_dict(meta, ['params', 'data'], k, formats)
    return meta


def arr_str(arr):
    return ' '.join(str(x) for x in arr)


def calc_recall(recall):
    recall = np.array(recall, dtype=np.float64)
    recall = np.round(recall * 100, 1)
    return recall

def calc_raw_recall(info):
    recalls = info.get('test_custom')
    if recalls is None:
        recalls = info['custom']
    return np.array(recalls, dtype=np.float64)

def recall_str(info):
    recalls = info.get('test_custom')
    if recalls is None:
        recalls = info['custom']
    recalls = calc_recall(recalls)
    if len(recalls) <= 5:
        rs = arr_str(recalls)
    else:
        rs = arr_str(recalls[:2]) + ' .. ' + arr_str(recalls[-2:])
    return rs


def main():
    arg_parser = argparse.ArgumentParser(description='View model info')
    arg_parser.add_argument(
        '--path',
        default=os.path.join('result', 'model_info.json'),
        help='Data file path'
    )
    arg_parser.add_argument('--all', action='store_true', help='Show all')
    arg_parser.add_argument(
        '--sort', default='wauc', help='sort by [default: %(default)s]'
    )
    arg_parser.add_argument('--avg')
    arg_parser.add_argument('--model')
    arg_parser.add_argument('--col-order', dest='col_order')
    arg_parser.add_argument('--data-cols', dest='data_cols')
    arg_parser.add_argument('--exclude-cols', dest='exclude_cols')

    args = arg_parser.parse_args()
    minfo = read_json(args.path)
    if args.all:
        infos = minfo['all']
    else:
        infos = [minfo['best']]
    if args.col_order:
        col_order = args.col_order.split(',')
    else:
        col_order = ['gauc', 'auc', 'factor_dim', 'reg_lambda', 'has_linear']
    exclude_cols = [] if args.exclude_cols is None else args.exclude_cols.split(',')
    exclude_cols = set(exclude_cols)
    formats = dict(
        wauc=dict(
            ndigits=4,
            width=10,
        ),
        test_score=dict(
            ndigits=4,
            width=10,
        ),
        score=dict(
            ndigits=4,
            width=10,
        ),
        auc=dict(
            ndigits=4,
        ),
        gauc=dict(
            ndigits=4,
        ),
        cnn_layer_sizes=dict(
            width=19,
        ),
        dnn_layer_sizes=dict(
            width=27,
        ),
    )
    meta = {}
    for sinfo in infos:
        fill_meta(
            meta,
            sinfo,
            formats=formats,
            data_cols=None if args.data_cols is None else args.data_cols.split(','),
            exclude_cols=exclude_cols
        )
    meta = list(meta.values())
    if col_order is not None:
        col_order = {c: i for i, c in enumerate(col_order)}
        meta.sort(key=lambda x: col_order.get(x['name'], 1000))

    meta = [
        get_meta_dict([], 'test_score', formats, title='tscore'),
        get_meta_dict([], 'score', formats)
    ] + meta

    if 'lr' not in exclude_cols:
        meta.append(dict(
            name='lr',
            proc=lambda info: info['params']['other'].get('learning_rates'),
            format=dict(
                # width=8,
                ndigits=5,
            ),
        ))

    if 'batch_size' not in exclude_cols:
        meta.append(dict(
            name='batch_size',
            proc=lambda info: info['params']['data'].get('batch_size'),
            format=dict(
                width=12,
            ),
        ))
    if 'check_size' not in exclude_cols:
        meta.append(dict(
            name='check_size',
            proc=lambda info: info['params']['data'].get('check_size'),
            format=dict(
                width=12,
            ),
        ))
    if args.model is not None and args.model == 'sfm':
        if 'vc' not in exclude_cols:
            meta.append(dict(
                name='vc',
                proc=lambda info: (
                    '??' if 'views' not in info['params']['model']
                    else len(info['params']['model'].get('views'))
                ),
                format=dict(
                    width=4,
                ),
            ))
    if 'data_path' not in exclude_cols:
        meta.append(dict(
            name='data_path',
            proc=lambda info: info['params']['data']['path'].rstrip('/').rsplit('/', 3)[1],
            # format=dict(
            #     width=10,
            # ),
        ))
    if 'recalls' not in exclude_cols:
        meta.append(dict(
            name='recalls',
            proc=lambda info: recall_str(info),
        ))
        meta.append(dict(
            name='raw_recalls',
            proc=lambda info: calc_raw_recall(info),
        ))

    df = pd.DataFrame(
        [[md['proc'](info) for md in meta] for info in infos],
        columns=[md['name'] for md in meta]
    )
    meta = [m for m in meta if m['name'] != 'raw_recalls']
    # print(meta)
    weights = gen_w()
    # print(weights, weights.dtype)
    # print(df['raw_recalls'].head())
    df['wauc'] = df['raw_recalls'].apply(lambda x: np.dot(weights, x))
    meta.insert(0, get_meta_dict([], 'wauc', formats))
    df.drop(['raw_recalls'], axis=1, inplace=True)
    # print(df)

    if args.avg:
        for col_name, dtype in df.dtypes.iteritems():
            if dtype == np.object_:
                df[col_name] = df[col_name].apply(lambda x: str(x))
        if args.avg == '_std':
            avg_by = [md['name'] for md in meta if md['name'] not in {'tscore', 'score'}]
        else:
            avg_by = args.avg.split(',')
        for col in df.columns:
            if col not in avg_by and col not in {'tscore', 'score'}:
                df[col] = np.nan
        # print(','.join(avg_by))
        df = df.groupby(avg_by).mean()
        df.reset_index(inplace=True)
        df.sort_values(by=['tscore'], ascending=True, inplace=True)
    else:
        if args.sort == '_std':
            sort_by = ['wauc']
        else:
            sort_by = args.sort.split(',')
        df.sort_values(by=sort_by, ascending=sort_by == ['tscore'], inplace=True)

    titles_class = namedtuple('Titles', [md['name'] for md in meta])
    # print(titles_class(*[md['name'] for md in meta]))
    print_info(meta, titles_class(*[md['name'] for md in meta]))
    for r in df.itertuples():
        # print(r)
        print_info(meta, r)
    test_metric = minfo.get('best_test_metric')
    if test_metric is not None:
        print('Test: ' + str(np.round(test_metric, 6)))


main()
