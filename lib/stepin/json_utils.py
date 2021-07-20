# encoding=utf-8

import gzip
import json


def read_json(path):
    with open(path) as f:
        return json.loads(f.read())


def write_json(path, obj, **kwargs):
    with open(path, 'w') as f:
        return f.write(json.dumps(obj, **kwargs))


def build_types_str(obj, indent=''):
    strings = []
    if isinstance(obj, (tuple, list)):
        for sobj in obj:
            strings.append(indent + build_types_str(sobj, indent=indent + '  '))
    elif isinstance(obj, dict):
        for k, sobj in obj.items():
            strings.append(indent + k)
            strings.append(build_types_str(sobj, indent=indent + '  '))
    else:
        strings.append(indent + str(type(obj)))
    return '\n'.join(strings)


def json_file_iter(path, gzipped=False):
    try:
        import ujson as json_
    except:
        json_ = json
    if gzipped:
        f = gzip.open(path, mode='rt')
    else:
        f = open(path)
    with f:
        for l in f:
            yield json_.loads(l)
