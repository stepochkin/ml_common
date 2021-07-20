import json

from sklearn.externals import joblib


def save_model(model, config, custom_meta_proc=None):
    joblib.dump(model, config['model_path'])
    with open(config['model_meta_path'], 'wb') as f:
        meta = dict()
        if custom_meta_proc is not None:
            custom_meta_proc(model, meta)
        json.dump(meta, f)


def load_model(config):
    model = joblib.load(config['model_path'])
    with open(config['model_meta_path'], 'rb') as f:
        meta = json.load(f)
    return model, meta
