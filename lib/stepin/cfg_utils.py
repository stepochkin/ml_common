# noinspection PyProtectedMember
from pydoc import locate


async def config2object(config):
    klass = locate(config['klass'])
    if klass is None:
        raise Exception('Class not found for path: ' + config['klass'])
    args = None
    try:
        args = config.get('args', {})
        obj_factory = klass(**args)
        obj = await obj_factory()
    except Exception as e:
        raise Exception('Error creating class %s instance with args %s' % (klass, args)) from e
    return obj


def extend_dict(d, **kwargs):
    d = d.copy()
    d.update(kwargs)
    return d


def extend_dict_0(d, new_params=None):
    if new_params is None:
        return d
    d = d.copy()
    d.update(new_params)
    return d


def update_dict(source_dict, target_dict):
    for k, v in source_dict.items():
        if isinstance(v, dict):
            tv = target_dict.get(k)
            if tv is None:
                tv = dict()
                target_dict[k] = tv
            update_dict(v, tv)
        else:
            target_dict[k] = v
