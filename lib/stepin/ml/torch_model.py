import json
import logging
import os

import torch

from stepin.json_utils import read_json
from stepin.ml.meta_train import TrainModel


class TorchModel(TrainModel):
    def __init__(
        self, model_factory, device='cpu',
        weight_decay=0.0, optimizer_params: dict=None
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
            if otype == 'adam':
                self.optimizer = torch.optim.Adam(
                    params=self.model.parameters(),
                    **oparams
                )
            elif otype == 'adagrad':
                self.optimizer = torch.optim.Adagrad(
                    params=self.model.parameters(),
                    **oparams
                )
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
        return dict(
            core_arguments=self.model_factory.get_params(),
        )

    def save(self, path, params_path=None):
        torch.save(self.model.state_dict(), path)
        if params_path is not None:
            with open(params_path, 'w') as f:
                f.write(json.dumps(self.get_params()))
        return path

    def restore(self, path):
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def create_from_files(cls, core_proc, params_path, vars_path):
        with open(params_path) as f:
            params = json.loads(f.read())
        core_arguments = params.pop('core_arguments', dict())
        cls.create_from_file(core_proc, params, core_arguments, vars_path)

    @classmethod
    def create_from_file(cls, core_proc, core_arguments, params, vars_path):
        class CoreFact(object):
            @staticmethod
            def create():
                return core_proc(core_arguments)

            @staticmethod
            def get_params():
                return core_arguments

        model = cls(CoreFact(), **params)
        model.restore(vars_path)
        return model

    def reset_target_value(self):
        self.target_value = 0.0

    def get_target_value(self):
        return self.target_value

    def ff(self, data):
        return self.model(data)

    def fit(self, data):
        self.model.train()
        loss = self.model_factory.calc_loss(self, data)
        # fields, target = data
        # fields, target = fields.to(self.device), target.to(self.device)
        # y = self.model(fields)
        # loss = self.criterion(y, target.float())
        self.model.zero_grad()
        loss.backward()
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

    def predict_iter(self, x, custom_proc=None, to_device_proc=None, tensor_cnt=1):
        if tensor_cnt == 1:
            output = []
        else:
            output = [[] for _ in range(tensor_cnt)]
        self.model.eval()
        with torch.no_grad():
            for bx in x:
                if to_device_proc is None:
                    bx = [t.to(self.device) for t in bx]
                else:
                    bx = to_device_proc(self, bx)
                if len(bx) == 1:
                    bx = bx[0]
                pred = self.model(bx)
                if custom_proc is not None:
                    pred = custom_proc(bx, pred)
                if tensor_cnt == 1:
                    output.append(pred)
                else:
                    for l, t in zip(output, pred):
                        l.append(t)
        if tensor_cnt == 1:
            return torch.cat(output, dim=0)
        return [torch.cat(l, dim=0) for l in output]


def load_model(model_path, params_or_path, core_from_params_proc, custom_core_params=None):
    if custom_core_params is None:
        custom_core_params = dict()
    params = read_json(params_or_path) if isinstance(params_or_path, str) else params_or_path
    core_arguments = params.pop('core_arguments', dict())
    core_arguments.update(custom_core_params)
    return TorchModel.create_from_file(
        core_from_params_proc, params, core_arguments, model_path
    )


def load_std_model(model_path, core_from_params_proc, **custom_core_params):
    return load_model(
        os.path.join(model_path, 'model.bin'),
        os.path.join(model_path, 'model_params.txt'),
        core_from_params_proc,
        custom_core_params=custom_core_params
    )
