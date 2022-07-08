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


class BestValues(object):
    def __init__(
        self, max_pass_count, max_worse_count,
        min_update_pass=0, update_step=1, max_warm_up_count=None,
        lr_max_worse_count=None, is_restore_prev_best=False, ce_min_decrease=0.0
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
        self.ce_min_decrease = ce_min_decrease
        self.cur_ce = None
        self.prev_lr_ce = None
        self.lr_ce = None
        self.ce = None
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

    def update(self, ce, cnn, custom=None):
        self.cur_ce = ce
        self.pass_count += 1
        if self.max_warm_up_count is not None:
            if self.pass_count == 1:
                self.warm_up_active = True
                logging.debug('Warming up activated')
                self.warm_up_loss = ce
            else:
                if self.warm_up_active and (self.warm_up_loss > ce or self.pass_count >= self.max_warm_up_count):
                    logging.debug('Warming up deactivated')
                    self.warm_up_active = False
                    self.worse_count = 0

        if self.best_score is None or ce < self.best_score:
            self.best_score = ce
        values = None
        if self.ce is None or ce < self.ce:
            self.ce = ce
            values = cnn.get_variable_values() if hasattr(cnn, 'get_variable_values') else None
            self.values = values

        if self.lr_ce is None or ce < (self.lr_ce - self.ce_min_decrease * (self.worse_count + 1)):
            logging.debug('Setting model checkpoint')
            self.prev_lr_ce = self.lr_ce
            self.lr_ce = ce
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
            self.cur_ce = None

    def restore_best(self, model):
        if self.cur_ce is not None and self.ce >= self.cur_ce:
            return
        if self.values is None:
            logging.debug('Cannot restore model values for score %s', self.ce)
        else:
            logging.debug('Restoring model values for score %s', self.ce)
            model.set_variable_values(self.values)
            self.cur_ce = self.ce

    def restore_prev_best(self, cnn):
        if self.prev_lr_values is None:
            logging.debug('Cannot restore model values for score %s', self.prev_lr_ce)
        else:
            logging.debug('Restoring model previous best values for score %s', self.prev_lr_ce)
            cnn.set_variable_values(self.prev_lr_values)
            self.cur_ce = self.prev_lr_ce

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
        lr = self._proc(self._pass_i, batch_i)
        self._set_lr(lr)

    def need_to_stop(self):
        if self._best_values.need_to_stop(0, True):
            return True
        return False


def multi_pass_train(
    model, train_iter_factory, valid_iter_factory, test_iter_factory,
    metric_calculator, best_values,
    learning_rates=None, before_train=None, show_custom_proc=None
):
    if learning_rates is None:
        learning_rates = [0.01]
    if callable(learning_rates):
        lr_strategy = ProcLrStrategy(best_values, model, learning_rates)
    else:
        lr_strategy = ListLrStrategy(best_values, model, learning_rates)
    has_target_value = hasattr(model, 'reset_target_value')

    def check_metric():
        if has_target_value:
            logging.info('Target = %s', model.get_target_value())
        if best_values.need_update():
            v = metric_calculator(model, valid_iter_factory)
            if isinstance(v, dict):
                metric = v['metric']
                custom = v.get('custom')
                loss = v.get('loss')
            else:
                metric = v
                custom = None
                loss = None
            if loss is not None and loss != 0.0:
                logging.info('Loss = %s', loss)
            logging.info('Score = %s', metric)
            if custom is not None:
                if show_custom_proc is None:
                    logging.info('%s', custom)
                else:
                    show_custom_proc(custom)
            best_values.update(metric, model, custom=custom)
        return lr_strategy.need_to_stop()

    pi = 0
    best_values.reset_worse_count()
    need_break = False
    checked = False
    batch_i = 0
    if before_train is not None:
        before_train()
    while True:
        logging.info('Pass %s', pi)
        lr_strategy.next_pass(pi)
        if has_target_value:
            model.reset_target_value()
        for batch in train_iter_factory():
            if isinstance(batch, MultiBatch):
                need_check = batch.need_check
                batch = batch.data
            else:
                need_check = False
            lr_strategy.next_batch(batch_i)
            batch_i += 1
            model.fit(batch)
            checked = False
            if need_check:
                checked = True
                if check_metric():
                    need_break = True
                    break
        pi += 1
        if need_break or (not checked and check_metric()):
            if lr_strategy.need_to_finish():
                break
    best_values.restore_best(model)
    if test_iter_factory is not None:
        tv = metric_calculator(model, test_iter_factory)
        if isinstance(tv, dict):
            tscore = tv['metric']
            tcustom = tv.get('custom')
        else:
            tscore = tv
            tcustom = None
        best_values.update_tscore(tscore, model, custom=tcustom)
        logging.info('Test metric = %s', tscore)
        if tcustom is not None:
            if show_custom_proc is None:
                logging.info('%s', tcustom)
            else:
                show_custom_proc(tcustom)


def _dict_product(d):
    keys = d.keys()
    for values in product(*tuple(d[k] for k in keys)):
        yield dict(zip(keys, values))


def _two_dict_product(d1, d2):
    keys1 = d1.keys()
    keys2 = d2.keys()
    for values in product(*tuple(d1[k] for k in keys1)):
        for values2 in product(*tuple(d2[k] for k in keys2)):
            yield dict(zip(keys1, values)), dict(zip(keys2, values2))


def _three_dict_product(d1, d2, d3):
    keys1 = d1.keys()
    keys2 = d2.keys()
    keys3 = d3.keys()
    for values in product(*tuple(d1[k] for k in keys1)):
        for values2 in product(*tuple(d2[k] for k in keys2)):
            for values3 in product(*tuple(d3[k] for k in keys3)):
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


def meta_train(
    config, model_factory, model_path, model_params_path, models_info_path,
    train_iter_factory, valid_iter_factory, metric_calculator,
    test_iter_factory=None, show_custom_proc=None, on_new_params=None,
    best_info_calculator=None, on_init_data=None
):
    last_infos = tuple()
    if os.path.exists(models_info_path):
        last_infos = read_json(models_info_path)['all']
    last_best_info = _find_best_info(last_infos)
    best_saved_info = last_best_info
    save_each_best_model = config.get('save_each_best_model', False)
    the_best_values = None
    the_best_model = None
    the_best_dconfig = None
    best_info = None
    m_infos = []
    repeat_count = config.get('repeat_count', 1)
    prev_dconfig = None
    for oconfig, dconfig, mconfig in _param_iter(config['train_params']):
        if on_new_params is not None:
            on_new_params(mconfig)
        all_config = dict(other=oconfig, data=dconfig, model=mconfig)
        if config.get('force_params'):
            rcount = repeat_count
        else:
            linfos = _find_infos(last_infos, all_config)
            rcount = max(0, repeat_count - len(linfos))
        if rcount > 0:
            logging.info('Training %s times', rcount)
        if (
            (on_init_data is not None) and (rcount > 0) and
            ((prev_dconfig is None) or (prev_dconfig != dconfig))
        ):
            try:
                on_init_data(dconfig, mconfig)
            except Exception as e:
                raise Exception(
                    'Error on config: ' + str(dconfig) + '  ' + str(mconfig)
                ) from e
        for repeat_i in range(rcount):
            model = model_factory(mconfig, dconfig)
            best_values = config_best_values(oconfig)
            logging.info('Fit %s ... %s', repeat_i + 1, all_config)
            learning_rates = oconfig.get('learning_rates', [0.01])
            if isinstance(learning_rates, str):
                learning_rates = eval(learning_rates)
            before_train = getattr(train_iter_factory, 'before_train', None)
            multi_pass_train(
                model,
                partial(train_iter_factory, dconfig),
                partial(valid_iter_factory, dconfig),
                None if test_iter_factory is None else partial(test_iter_factory, dconfig),
                metric_calculator, best_values, learning_rates=learning_rates,
                before_train=before_train,
                show_custom_proc=show_custom_proc
            )
            # model.fit(_multi_pass_iter(lambda: train_iter_factory(dconfig), metric_calculator, model, best_values))
            info = dict(
                params=all_config, score=best_values.lr_ce,
                test_score=best_values.tscore
            )
            if best_values.custom is not None:
                info['custom'] = best_values.custom
            if best_values.tcustom is not None:
                info['test_custom'] = best_values.tcustom
            m_infos.append(info)
            best_values.clean_cache()
            if hasattr(model, 'to_cpu'):
                model.to_cpu()
            if the_best_values is None or best_values.best_score < the_best_values.best_score:
                the_best_values = best_values
                the_best_model = model
                the_best_dconfig = dconfig
                best_info = info
            all_infos = []
            all_infos.extend(m_infos)
            all_infos.extend(last_infos)
            ensure_parent_path(models_info_path)
            write_json(
                models_info_path,
                dict(
                    all=all_infos,
                    best=_find_best_info(all_infos),
                )
            )
            if save_each_best_model and (model_path is not None):
                if need_to_save(best_saved_info, best_info):
                    ensure_parent_path(model_path)
                    ensure_parent_path(model_params_path)
                    save_path = the_best_model.save(
                        model_path, params_path=model_params_path,
                        dconfig=the_best_dconfig
                    )
                    logging.debug("Model with score %s saved to file: %s", the_best_values.best_score, save_path)
                    best_saved_info = best_info
                    if best_info_calculator is not None:
                        best_info_calculator(the_best_model, lambda: valid_iter_factory(the_best_dconfig))
        prev_dconfig = dconfig

    if not save_each_best_model and (model_path is not None):
        if need_to_save(best_saved_info, best_info):
            ensure_parent_path(model_path)
            ensure_parent_path(model_params_path)
            save_path = the_best_model.save(
                model_path, params_path=model_params_path,
                dconfig=the_best_dconfig
            )
            logging.debug("Model with score %s saved to file: %s", the_best_values.best_score, save_path)
            if best_info_calculator is not None:
                best_info_calculator(the_best_model, lambda: valid_iter_factory(the_best_dconfig))


def config_best_values(oconfig):
    best_values = BestValues(
        oconfig.get('pass_count', 1),
        oconfig.get('stop_worse_count', 1),
        min_update_pass=oconfig.get('min_update_pass', 0) - 1,
        update_step=oconfig.get('update_step', 1),
        max_warm_up_count=oconfig.get('warm_up_count', None),
        lr_max_worse_count=oconfig.get('lr_stop_worse_count', 1),
        is_restore_prev_best=oconfig.get('is_restore_prev_best', False),
        ce_min_decrease=oconfig.get('metric_min_decrease', 0.0)
    )
    return best_values
