import abc
import logging
from collections import namedtuple
from functools import partial
from itertools import product
import os

from stepin.file_utils import ensure_parent_path
from stepin.json_utils import write_json, read_json


class TrainModel(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def save(self, path, params_path=None):
        pass

    @abc.abstractmethod
    def restore(self, path):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass


class BestValues:
    def __init__(
        self, max_pass_count, max_worse_count,
        min_update_pass=0, update_step=1, max_warm_up_count=None,
        lr_max_worse_count=None, is_restore_prev_best=False, score_min_decrease=0.0
    ):
        self.max_pass_count = max_pass_count
        self.max_worse_count = max_worse_count
        self.lr_max_worse_count = 1 if lr_max_worse_count is None else lr_max_worse_count
        self.is_restore_prev_best = is_restore_prev_best
        self.min_update_pass = min_update_pass
        self.update_step = update_step
        self.max_warm_up_count = max_warm_up_count
        self.warm_up_active = False
        self.warm_up_loss = None
        self.pass_count = 0
        self.score_min_decrease = score_min_decrease
        self.cur_score = None
        self.prev_lr_score = None
        self.lr_score = None
        self.score = None
        self.custom = None
        self.tscore = None
        self.best_score = None
        self.tcustom = None
        self.tvalues = None
        self.prev_lr_values = None
        self.lr_values = None
        self.values = None
        self.worse_count = 0

    def clean_cache(self):
        self.tvalues = None
        self.prev_lr_values = None
        self.lr_values = None
        self.values = None

    def need_update(self):
        if self.pass_count < self.min_update_pass:
            self.pass_count += 1
            return False
        if (self.pass_count - self.min_update_pass) % self.update_step != 0:
            self.pass_count += 1
            return False
        return True

    def update(self, score, cnn, custom=None):
        self.cur_score = score
        self.pass_count += 1
        if self.max_warm_up_count is not None:
            if self.pass_count == 1:
                self.warm_up_active = True
                logging.debug('Warming up activated')
                self.warm_up_loss = score
            else:
                if self.warm_up_active and (self.warm_up_loss > score or self.pass_count >= self.max_warm_up_count):
                    logging.debug('Warming up deactivated')
                    self.warm_up_active = False
                    self.worse_count = 0

        if self.best_score is None or score < self.best_score:
            self.best_score = score
        values = None
        if self.score is None or score < self.score:
            self.score = score
            values = cnn.get_variable_values() if hasattr(cnn, 'get_variable_values') else None
            self.values = values

        if self.lr_score is None or score < (self.lr_score - self.score_min_decrease * (self.worse_count + 1)):
            logging.debug('Setting model checkpoint')
            self.prev_lr_score = self.lr_score
            self.lr_score = score
            self.prev_lr_values = self.lr_values
            if values is None:
                self.lr_values = cnn.get_variable_values() if hasattr(cnn, 'get_variable_values') else None
            else:
                self.lr_values = values
            self.custom = custom
            self.worse_count = 0
        else:
            self.worse_count += self.update_step

    def update_tscore(self, score, model, custom=None):
        if self.tscore is None or score < self.tscore:
            self.tscore = score
            self.tvalues = model.get_variable_values() if hasattr(model, 'get_variable_values') else None
            self.tcustom = custom
        if self.best_score is None or score < self.best_score:
            self.best_score = score

    def reset_worse_count(self):
        self.worse_count = 0

    def restore_tbest(self, model):
        if self.tvalues is None:
            logging.debug('Cannot restore model values for tscore %s', self.tscore)
        else:
            logging.debug('Restoring model values for tscore %s', self.tscore)
            model.set_variable_values(self.tvalues)
            self.cur_score = None

    def restore_best(self, model):
        if self.cur_score is not None and self.score >= self.cur_score:
            return
        if self.values is None:
            logging.debug('Cannot restore model values for score %s', self.score)
        else:
            logging.debug('Restoring model values for score %s', self.score)
            model.set_variable_values(self.values)
            self.cur_score = self.score

    def restore_prev_best(self, cnn):
        if self.prev_lr_values is None:
            logging.debug('Cannot restore model values for score %s', self.prev_lr_score)
        else:
            logging.debug('Restoring model previous best values for score %s', self.prev_lr_score)
            cnn.set_variable_values(self.prev_lr_values)
            self.cur_score = self.prev_lr_score

    def lr_finished(self, model, is_last_lr):
        if not is_last_lr:
            if self.is_restore_prev_best:
                self.restore_prev_best(model)
            else:
                self.restore_best(model)

    def need_to_stop(self, learning_rate_index, is_last_learning_rate):
        # logging.debug(
        #     'pass_count = %s, worse_count = %s, is_last_learning_rate = %s'
        #     ', max_worse_count = %s, lr_max_worse_count = %s',
        #     self.pass_count, self.worse_count, is_last_learning_rate, self.max_worse_count, self.lr_max_worse_count
        # )
        if self.pass_count >= self.max_pass_count:
            return True
        if self.pass_count < self.min_update_pass:
            return False
        if self.warm_up_active:
            return False
        if isinstance(self.lr_max_worse_count, (list, tuple)):
            if learning_rate_index < len(self.lr_max_worse_count):
                return self.worse_count >= self.lr_max_worse_count[learning_rate_index]
            else:
                return self.worse_count >= self.max_worse_count
        else:
            if is_last_learning_rate:
                return self.worse_count >= self.max_worse_count
            else:
                return self.worse_count >= self.lr_max_worse_count


MultiBatch = namedtuple('MultiBatch', ['data', 'need_check'])

class LrStrategy:
    def __init__(self, model, best_values):
        self._model = model
        self._best_values = best_values

    def _set_lr(self, learning_rate):
        logging.info('Learning rate %s', learning_rate)
        self._model.set_learning_rate(learning_rate)

    def next_pass(self, pass_i):
        pass

    def next_batch(self, batch_i):
        pass

    def need_to_stop(self):
        return True

    def need_to_finish(self):
        return True

class ListLrStrategy(LrStrategy):
    def __init__(self, best_values, model, learning_rates):
        super().__init__(model, best_values)
        self.learning_rates = learning_rates
        self.lr_ind = 0
        self._set_lr(self.learning_rates[self.lr_ind])

    def next_pass(self, pass_i):
        pass

    def next_batch(self, batch_i):
        pass

    def need_to_stop(self):
        print(self.lr_ind, self.learning_rates)
        is_last_learning_rate = self.lr_ind >= len(self.learning_rates) - 1
        if self._best_values.need_to_stop(self.lr_ind, is_last_learning_rate):
            self._best_values.lr_finished(self._model, is_last_learning_rate)
            self.lr_ind += 1
            if not is_last_learning_rate:
                self._set_lr(self.learning_rates[self.lr_ind])
                self._best_values.reset_worse_count()
            return True
        return False

    def need_to_finish(self):
        return self.lr_ind >= len(self.learning_rates)


class ProcLrStrategy(LrStrategy):
    def __init__(self, best_values, model, proc):
        super().__init__(model, best_values)
        self._proc = proc
        self._pass_i = None

    def next_pass(self, pass_i):
        self._pass_i = pass_i

    def next_batch(self, batch_i):
        learning_rate = self._proc(self._pass_i, batch_i)
        self._set_lr(learning_rate)

    def need_to_stop(self):
        if self._best_values.need_to_stop(0, True):
            return True
        return False


def _dict_product(dict_):
    keys = dict_.keys()
    for values in product(*tuple(dict_[k] for k in keys)):
        yield dict(zip(keys, values))


def _two_dict_product(dict1, dict2):
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    for values in product(*tuple(dict1[k] for k in keys1)):
        for values2 in product(*tuple(dict2[k] for k in keys2)):
            yield dict(zip(keys1, values)), dict(zip(keys2, values2))


def _three_dict_product(dict1, dict2, dict3):
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    keys3 = dict3.keys()
    for values in product(*tuple(dict1[k] for k in keys1)):
        for values2 in product(*tuple(dict2[k] for k in keys2)):
            for values3 in product(*tuple(dict3[k] for k in keys3)):
                yield dict(zip(keys1, values)), dict(zip(keys2, values2)), dict(zip(keys3, values3))


def _find_info(infos, params):
    for info in infos:
        if info['params'] == params:
            return info
    return None


def _find_infos(infos, params):
    result = []
    for info in infos:
        if info['params'] == params:
            result.append(info)
    return result


def _find_best_info(infos):
    best_info = None
    for info in infos:
        if need_to_save(best_info, info):
            best_info = info
    return best_info


def _param_iter(tconfigs):
    if isinstance(tconfigs, dict):
        tconfigs = [tconfigs]
    for tconfig in tconfigs:
        oconfigs = tconfig['other']
        if isinstance(oconfigs, dict):
            oconfigs = [oconfigs]
        mconfigs = tconfig['model']
        if isinstance(mconfigs, dict):
            mconfigs = [mconfigs]
        for odict in oconfigs:
            for mdict in mconfigs:
                for oconfig, dconfig, mconfig in _three_dict_product(odict, tconfig.get('data', {}), mdict):
                    yield oconfig, dconfig, mconfig


def need_to_save(best_saved_info, best_info):
    if best_saved_info is None:
        return True
    has_test_score = (
        best_info.get('test_score') is not None and
        best_saved_info.get('test_score') is not None
    )
    if has_test_score:
        return best_info['test_score'] < best_saved_info['test_score']
    return best_info['score'] < best_saved_info['score']

class Trainer:
    def __init__(self, config, model_factory, train_iter_factory, valid_iter_factory):
        self.config = config
        self.model_factory = model_factory
        self.model_path = None
        self.model_params_path = None
        self.models_info_path = None
        self.train_iter_factory = train_iter_factory
        self.valid_iter_factory = valid_iter_factory
        self.metric_calculator = None
        self.test_iter_factory = None
        self.show_custom_proc = None
        self.on_new_params = None
        self.best_info_calculator = None

        self.best_saved_info = None
        self.the_best_values = None
        self.the_best_model = None
        self.the_best_dconfig = None
        self.best_values = None
        self.model = None
        self.dconfig = None

    def set_paths(self, model_path, model_params_path, models_info_path):
        self.model_path = model_path
        self.model_params_path = model_params_path
        self.models_info_path = models_info_path

    def set_metric_calculator(self, value):
        self.metric_calculator = value

    def set_test_iter_factory(self, value):
        self.test_iter_factory = value

    def set_show_custom_proc(self, value):
        self.show_custom_proc = value

    def set_on_new_params(self, value):
        self.on_new_params = value

    def set_best_info_calculator(self, value):
        self.best_info_calculator = value

    def meta_train(self):
        last_infos = tuple()
        if (self.models_info_path is not None) and os.path.exists(self.models_info_path):
            last_infos = read_json(self.models_info_path)['all']
        last_best_info = _find_best_info(last_infos)
        self.best_saved_info = last_best_info
        self.the_best_values = None
        self.the_best_model = None
        self.the_best_dconfig = None
        best_info = None
        m_infos = []
        repeat_count = self.config.get('repeat_count', 1)
        for oconfig, dconfig, mconfig in _param_iter(self.config['train_params']):
            self.dconfig = dconfig
            if self.on_new_params is not None:
                self.on_new_params(mconfig)
            all_config = dict(other=oconfig, data=dconfig, model=mconfig)
            if self.config.get('force_params'):
                rcount = repeat_count
            else:
                linfos = _find_infos(last_infos, all_config)
                rcount = max(0, repeat_count - len(linfos))
            logging.info('Training %s times', rcount)
            for repeat_i in range(rcount):
                model = self.model_factory(mconfig, dconfig)
                self.model = model
                best_values = config_best_values(oconfig)
                self.best_values = best_values
                logging.info('Fit %s ... %s', repeat_i + 1, all_config)
                learning_rates = oconfig.get('learning_rates', [0.01])
                if isinstance(learning_rates, str):
                    learning_rates = eval(learning_rates)
                self.multi_pass_train(learning_rates)
                info = self._build_info(all_config)
                m_infos.append(info)
                best_values.clean_cache()
                if hasattr(model, 'to_cpu'):
                    model.to_cpu()
                if self.the_best_values is None or best_values.best_score < self.the_best_values.best_score:
                    self.the_best_values = best_values
                    self.the_best_model = model
                    self.the_best_dconfig = dconfig
                    best_info = info
                all_infos = []
                all_infos.extend(m_infos)
                all_infos.extend(last_infos)
                if self.models_info_path is not None:
                    ensure_parent_path(self.models_info_path)
                    write_json(
                        self.models_info_path,
                        dict(
                            all=all_infos,
                            best=_find_best_info(all_infos),
                        )
                    )
                self._save_model(best_info)
        self._save_model(best_info)

    def _build_info(self, all_config):
        info = dict(
            params=all_config, score=self.best_values.lr_score,
            test_score=self.best_values.tscore
        )
        if self.best_values.custom is not None:
            info['custom'] = self.best_values.custom
        if self.best_values.tcustom is not None:
            info['test_custom'] = self.best_values.tcustom
        return info

    def _save_model(self, best_info):
        save_each_best_model = self.config.get('save_each_best_model', False)
        if save_each_best_model and (self.model_path is not None):
            if need_to_save(self.best_saved_info, best_info):
                ensure_parent_path(self.model_path)
                ensure_parent_path(self.model_params_path)
                save_path = self.the_best_model.save(
                    self.model_path, params_path=self.model_params_path,
                    dconfig=self.the_best_dconfig
                )
                logging.debug(
                    "Model with score %s saved to file: %s",
                    self.the_best_values.best_score, save_path
                )
                self.best_saved_info = best_info
                if self.best_info_calculator is not None:
                    self.best_info_calculator(
                        self.the_best_model,
                        lambda: self.valid_iter_factory(self.the_best_dconfig)
                    )

    def multi_pass_train(self, learning_rates):
        if callable(learning_rates):
            lr_strategy = ProcLrStrategy(self.best_values, self.model, learning_rates)
        else:
            lr_strategy = ListLrStrategy(self.best_values, self.model, learning_rates)
        has_target_value = hasattr(self.model, 'reset_target_value')

        pass_i = 0
        self.best_values.reset_worse_count()
        need_break = False
        checked = False
        batch_i = 0
        before_train = getattr(self.train_iter_factory, 'before_train', None)
        if before_train is not None:
            before_train()
        while True:
            logging.info('Pass %s', pass_i)
            lr_strategy.next_pass(pass_i)
            if has_target_value:
                self.model.reset_target_value()
            for batch in self.train_iter_factory(self.dconfig):
                if isinstance(batch, MultiBatch):
                    need_check = batch.need_check
                    batch = batch.data
                else:
                    need_check = False
                lr_strategy.next_batch(batch_i)
                batch_i += 1
                self.model.fit(batch)
                checked = False
                if need_check:
                    checked = True
                    if self.check_metric(lr_strategy):
                        need_break = True
                        break
            pass_i += 1
            if need_break or (not checked and self.check_metric(lr_strategy)):
                if lr_strategy.need_to_finish():
                    break
        self.best_values.restore_best(self.model)
        self._run_test()

    def _run_test(self):
        if self.test_iter_factory is None:
            return
        metrics = self.metric_calculator(
            self.model, partial(self.test_iter_factory, self.dconfig)
        )
        if isinstance(metrics, dict):
            tscore = metrics['metric']
            tcustom = metrics.get('custom')
        else:
            tscore = metrics
            tcustom = None
        self.best_values.update_tscore(tscore, self.model, custom=tcustom)
        logging.info('Test metric = %s', tscore)
        if tcustom is not None:
            if self.show_custom_proc is None:
                logging.info('%s', tcustom)
            else:
                self.show_custom_proc(tcustom)

    def check_metric(self, lr_strategy):
        has_target_value = hasattr(self.model, 'reset_target_value')
        if has_target_value:
            logging.info('Target = %s', self.model.get_target_value())
        if self.best_values.need_update():
            metrics = self.metric_calculator(
                self.model, partial(self.valid_iter_factory, self.dconfig)
            )
            if isinstance(metrics, dict):
                metric = metrics['metric']
                custom = metrics.get('custom')
                loss = metrics.get('loss')
            else:
                metric = metrics
                custom = None
                loss = None
            if loss is not None and loss != 0.0:
                logging.info('Loss = %s', loss)
            logging.info('Score = %s', metric)
            if custom is not None:
                if self.show_custom_proc is None:
                    logging.info('%s', custom)
                else:
                    self.show_custom_proc(custom)
            self.best_values.update(metric, self.model, custom=custom)
        return lr_strategy.need_to_stop()


def config_best_values(oconfig):
    best_values = BestValues(
        oconfig.get('pass_count', 1),
        oconfig.get('stop_worse_count', 1),
        min_update_pass=oconfig.get('min_update_pass', 0) - 1,
        update_step=oconfig.get('update_step', 1),
        max_warm_up_count=oconfig.get('warm_up_count', None),
        lr_max_worse_count=oconfig.get('lr_stop_worse_count', 1),
        is_restore_prev_best=oconfig.get('is_restore_prev_best', False),
        score_min_decrease=oconfig.get('metric_min_decrease', 0.0)
    )
    return best_values
