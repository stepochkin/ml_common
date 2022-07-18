from abc import ABC, abstractmethod
from typing import Iterable

from stepin.ml.meta_train import MultiBatch, meta_train
from stepin.ml.recs_metrics import calc_batch_precision_recall_ndcg
from stepin.ml.torch_model import TorchModel


class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config

    def on_init_data(self, dconfig, mconfig):
        pass

    @abstractmethod
    def model_factory(self, mconfig) -> TorchModel:
        pass

    @abstractmethod
    def train_data_factory(self, dconfig, mconfig) -> Iterable:
        pass

    @abstractmethod
    def valid_data_factory(self, dconfig, mconfig) -> Iterable:
        pass

    def test_data_factory(self, dconfig, mconfig) -> Iterable:
        pass

    @abstractmethod
    def metric_calc(self, model, data_iter_factory) -> dict:
        pass

    @staticmethod
    def _check_size_iter(config, batch_iter):
        check_size = config.get('check_size')
        if check_size is None:
            for batch in batch_iter:
                yield batch[:-1]
            return
        rows_count = 0
        for batch in batch_iter:
            rows_count += batch[-1]
            need_check = rows_count >= check_size
            yield MultiBatch(batch[:-1], need_check)
            if need_check:
                rows_count = 0

    @staticmethod
    def _prn_metrics(recs, positives):
        precs, recalls, ndcg = calc_batch_precision_recall_ndcg(recs, positives)

        return dict(
            metric=-float(ndcg[-1]),
            custom=dict(
                ndcg=float(ndcg[-1]),
                p=float(precs[-1]),
                r=float(recalls[-1]),
            ),
        )

    def train_and_save(
        self,
        test_iter_factory=None, show_custom_proc=None, on_new_params=None,
        best_info_calculator=None
    ):
        meta_train(
            self.config, self.model_factory,
            self.config['model_path'], self.config['model_params_path'],
            self.config['model_info_path'],
            self.train_data_factory, self.valid_data_factory, self.metric_calc,
            test_iter_factory=test_iter_factory,
            show_custom_proc=show_custom_proc,
            on_new_params=on_new_params,
            best_info_calculator=best_info_calculator,
            on_init_data=self.on_init_data
        )

    def train_and_return(
        self,
        test_iter_factory=None, show_custom_proc=None, on_new_params=None,
        best_info_calculator=None
    ):
        return meta_train(
            self.config, self.model_factory,
            None, None, None,
            self.train_data_factory, self.valid_data_factory, self.metric_calc,
            test_iter_factory=test_iter_factory,
            show_custom_proc=show_custom_proc,
            on_new_params=on_new_params,
            best_info_calculator=best_info_calculator,
            on_init_data=self.on_init_data
        )


class TrainerData(ABC):
    @abstractmethod
    def init(self):
        pass


class BaseDpTrainer(BaseTrainer, ABC):
    def __init__(self, config, data: TrainerData):
        super().__init__(config)
        self.data: TrainerData = data

    def on_init_data(self, dconfig, mconfig):
        self.data.init()
