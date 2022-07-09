import abc


class TrainModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def save(self, path, params_path=None, dconfig=None):
        pass

    @abc.abstractmethod
    def restore(self, path):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass
