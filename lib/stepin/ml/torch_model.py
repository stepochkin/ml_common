import json
import os

import numpy as np
import torch

from stepin.json_utils import read_json
from stepin.ml.meta_train import TrainModel


class TorchModel(TrainModel):
    def __init__(
        self, model_factory, device='cpu',
        weight_decay=0.0, optimizer_params: dict = None
    ):
        super(TorchModel, self).__init__()
        self.device = torch.device(device)
        self.model_factory = model_factory
        self.model: torch.nn.Module = self.model_factory.create()
        self.model.to(self.device)
        if optimizer_params is None:
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                weight_decay=weight_decay
            )
        else:
            oparams = optimizer_params.copy()
            otype = oparams.pop('type', 'adam').lower()
            factory = oparams.pop('factory', None)
            if factory is not None:
                params = factory(self.model, oparams)
            else:
                params = dict(
                    params=self.model.parameters(),
                    **oparams
                )
            print(params)
            if otype == 'adam':
                self.optimizer = torch.optim.Adam(**params)
            elif otype == 'adagrad':
                self.optimizer = torch.optim.Adagrad(**params)
            else:
                raise Exception('Unknown optimizer ' + otype)
        self.target_value = 0.0
        self.steps = 0

    def __del__(self):
        if self.model is not None:
            del self.model
            self.model = None

    def to_device(self, tensor):
        return tensor.to(self.device)

    def to_cpu(self):
        if self.model is not None:
            # r = torch.cuda.memory_reserved(0)
            # a = torch.cuda.memory_allocated(0)
            # logging.debug('CUDA: reserved %s allocated %s', r, a)

            self.model = self.model.to(torch.device('cpu'))

            # r = torch.cuda.memory_reserved(0)
            # a = torch.cuda.memory_allocated(0)
            # logging.debug('CUDA: reserved %s allocated %s', r, a)

    def get_variable_values(self):
        return [params.clone() for params in self.model.parameters()]

    def set_variable_values(self, values):
        for params, vs in zip(self.model.parameters(), values):
            params.data.copy_(vs)

    def set_learning_rate(self, learning_rate):
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

    def get_learning_rate(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def get_params(self):
        return self.model_factory.get_params()

    def save(self, path, params_path=None, dconfig=None):
        torch.save(self.model.state_dict(), path)
        if params_path is not None:
            params = self.get_params()
            if dconfig is not None:
                params['dconfig'] = dconfig
            with open(params_path, 'w') as f:
                f.write(json.dumps(params))
        return path

    def restore(self, path):
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def create_from_files(cls, model_cls, params_path, vars_path):
        with open(params_path) as f:
            params = json.loads(f.read())
            if 'dconfig' in params:
                del params['dconfig']
        return cls.create_from_file(model_cls, params, vars_path)

    @classmethod
    def create_from_file(cls, model_cls, params, vars_path):
        class CoreFact(object):
            @staticmethod
            def create():
                return model_cls(**params)

            @staticmethod
            def get_params():
                return params

        model = cls(CoreFact())
        model.restore(vars_path)
        return model

    def reset_target_value(self):
        self.target_value = 0.0

    def get_target_value(self):
        return self.target_value

    def ff(self, *data):
        return self.model(*data)

    def fit(self, data):
        self.model.train()
        loss = self.model_factory.calc_loss(self, *data)
        # fields, target = data
        # fields, target = fields.to(self.device), target.to(self.device)
        # y = self.model(fields)
        # loss = self.criterion(y, target.float())
        self.model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.target_value += float(loss.item())
        self.steps += 1

    def loss(self, data, is_train):
        self.model.train(mode=is_train)
        return self.model_factory.calc_loss(self, data)

    def loss_iter(self, it, is_train):
        self.model.train(mode=is_train)
        loss = 0.0
        n = 0
        for bx in it:
            loss += self.model_factory.calc_loss(self, bx).item()
            n += 1
        return loss / n

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def list_struct_to_device(self, struct):
        return tuple(
            self.list_struct_to_device(t) if isinstance(t, (tuple, list))
            else (t.to(self.device) if torch.is_tensor(t) else t)
            for t in struct
        )

    def struct_to_device(self, struct):
        if torch.is_tensor(struct):
            return struct.to(self.device)
        if isinstance(struct, dict):
            return {n: self.struct_to_device(t) for n, t in struct.items()}
        if isinstance(struct, (tuple, list)):
            return tuple(self.struct_to_device(t) for t in struct)
        return struct

    def predict_iter(
        self, x, call_proc=None, custom_proc=None,
        to_device_proc=None, tensor_cnt=1
    ):
        if tensor_cnt == 1:
            output = []
        else:
            output = [[] for _ in range(tensor_cnt)]
        self.model.eval()
        with torch.no_grad():
            for bx in x:
                if to_device_proc is None:
                    bx = self.struct_to_device(bx)

                    # if isinstance(bx, dict):
                    #     bx = {
                    #         n: t.to(self.device) if torch.is_tensor(t) else t
                    #         for n, t in bx.items()
                    #     }
                    # else:
                    #     bx = self.list_struct_to_device(bx)
                else:
                    bx = to_device_proc(self, bx)
                # if len(bx) == 1:
                #     bx = bx[0]
                if call_proc is None:
                    if isinstance(bx, dict):
                        pred = self.model(**bx)
                    else:
                        pred = self.model(*bx)
                else:
                    pred = call_proc(self.model, bx)
                pred = (
                    [t.cpu().numpy() for t in pred]
                    if isinstance(pred, (tuple, list))
                    else pred.cpu().numpy()
                )
                if custom_proc is not None:
                    pred = custom_proc(bx, pred)
                if tensor_cnt == 1:
                    output.append(pred)
                else:
                    for tl, t in zip(output, pred):
                        tl.append(t)
        if tensor_cnt == 1:
            return np.vstack(output)
        return [np.vstack(tl) for tl in output]


def load_model(model_cls, model_path, params_or_path):
    params = read_json(params_or_path) if isinstance(params_or_path, str) else params_or_path
    params.pop('dconfig')
    return TorchModel.create_from_file(model_cls, params, model_path)


def load_std_model(model_cls, model_path):
    return load_model(
        model_cls,
        os.path.join(model_path, 'model.bin'),
        os.path.join(model_path, 'model_params.txt')
    )
