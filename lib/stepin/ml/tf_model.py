import abc
import json
import logging
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

from stepin.json_utils import read_json
from stepin.ml.meta_train import TrainModel
from stepin.ml.yellowfin import YFOptimizer


class TfCore(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_learning_rate(self, session):
        pass

    @abc.abstractmethod
    def set_learning_rate(self, session, learning_rate):
        pass

    @abc.abstractmethod
    def build_graph(self, g):
        pass

    @abc.abstractmethod
    def get_init_all_vars(self):
        pass

    @abc.abstractmethod
    def build_train_fd(self, x):
        pass

    @abc.abstractmethod
    def get_minimizer(self):
        pass

    @abc.abstractmethod
    def get_gradients(self):
        pass

    @abc.abstractmethod
    def get_loss(self):
        pass

    @abc.abstractmethod
    def get_predict(self):
        pass

    @abc.abstractmethod
    def get_summary_op(self):
        pass

    @abc.abstractmethod
    def build_loss_fd(self, x, is_train):
        pass

    @abc.abstractmethod
    def build_predict_fd(self, x):
        pass


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
    return list(zip(clipped_gradients, variables))


# noinspection PyAttributeOutsideInit,PyAbstractClass
class AbstractTfCore(TfCore):
    def __init__(self, device='gpu', clip_gradients=None):
        self.clip_gradients = clip_gradients
        self.device = device
        self.optimizer_name = 'adam'
        self.optimizer_params = dict()
        self.dtype_bool = tf.float32 if device == 'gpu' else tf.bool
        self.dtype_int = tf.float32 if device == 'gpu' else tf.int32

    def set_learning_rate(self, session, lr):
        session.run(self.learning_rate_setter, feed_dict={self.learning_rate_ph: lr})

    def get_learning_rate(self, session):
        # noinspection PyProtectedMember,PyUnresolvedReferences
        return session.run(self.optimizer._lr_t)

    def set_clip_gradients(self, value):
        self.clip_gradients = value
        return self

    def set_device(self, value):
        self.device = value
        return self

    def set_optimizer_params(self, value):
        self.optimizer_name = value.pop('name', 'adam').lower()
        self.optimizer_params = value
        return self

    def dev_bool(self, t):
        if self.device == 'gpu':
            return tf.cast(t, tf.bool)
        return t

    def dev_int(self, t):
        if self.device == 'gpu':
            return tf.cast(t, tf.int32)
        return t

    def dev_float(self, t):
        if self.device == 'gpu':
            return tf.cast(t, tf.float32)
        return t

    def summary_scalar(self, message, scalar):
        if self.device != 'gpu':
            tf.summary.scalar(message, scalar)

    @abc.abstractmethod
    def do_build_graph(self, g):
        pass

    def get_gradients(self):
        return None

    def _build_graph(self, g):
        self.do_build_graph(g)
        self.learning_rate_var = tf.Variable(0.001, trainable=False, name='learning_rate')
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate_ph')
        self.learning_rate_setter = tf.assign(self.learning_rate_var, self.learning_rate_ph)
        if self.optimizer_name == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_var, **self.optimizer_params)
        elif self.optimizer_name == 'yellow_fin':
            self.optimizer = YFOptimizer(learning_rate=self.learning_rate_var, **self.optimizer_params)
        else:
            raise Exception('Invalid optimizer_name ' + self.optimizer_name)
        grads = self.get_gradients()
        if grads is None:
            loss = self.get_loss()
            grads = self.optimizer.compute_gradients(loss)
        if self.clip_gradients is not None:
            # self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.optimizer, self.clip_gradients)
            grads = _clip_gradients_by_norm(grads, self.clip_gradients)

        # init_lr = 0.001
        # global_step = tf.Variable(0)
        # learning_rate = tf.train.exponential_decay(
        #     init_lr, global_step, decay_steps=500, decay_rate=0.95,
        #     staircase=True
        # )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.minimizer = self.optimizer.apply_gradients(grads)

        self.init_all_vars = tf.global_variables_initializer()
        self.summary_op = tf.summary.merge_all()

    def build_graph(self, g):
        if self.device.startswith('/'):
            device_name = self.device
        else:
            device_name = '/' + self.device + ':0'
        # logging.debug('device_name = %s', device_name)
        with g.device(device_name):
            self._build_graph(g)

    def get_minimizer(self):
        return self.minimizer

    def get_init_all_vars(self):
        return self.init_all_vars

    def get_summary_op(self):
        return self.summary_op


class TfModel(TrainModel):
    def __init__(
        self, core_factory,
        log_dir=None, session_config=None, verbose=0,
        seed=None, debug_path=None
    ):
        super(TfModel, self).__init__()
        self.g = tf.Graph()
        self.session = None
        self.core_factory = core_factory
        self.core = self.core_factory.create()
        self.need_logs = log_dir is not None
        self.log_dir = log_dir
        self.session_config_dict = session_config
        if session_config is None:
            self.session_config = None
        else:
            self.session_config = tf.ConfigProto(**session_config)
        self.verbose = verbose
        self.steps = 0
        self.seed = seed
        self.debug_path = debug_path
        self.target_value = None
        self.summary_writer = None

    def get_variables(self):
        return self.g.get_collection(tf_ops.GraphKeys.TRAINABLE_VARIABLES)

    def get_variable_values(self):
        self.ensure_session()
        return self.session.run(self.get_variables())

    def get_variable_dict(self, regexp=None):
        if isinstance(regexp, str):
            regexp = re.compile(regexp)
        variables = self.get_variables()
        if regexp is not None:
            variables = [v for v in variables if regexp.match(v.name) is not None]
        self.ensure_session()
        values = self.session.run(variables)
        return {var.name: val for var, val in zip(variables, values)}

    def set_variable_values(self, values):
        self.ensure_session()
        self.session.run([v.assign(vv) for v, vv in zip(self.get_variables(), values)])

    def get_variable_value(self, name):
        self.ensure_session()
        return self.session.run(self.find_var(name))

    def find_var(self, name):
        for v in self.get_variables():
            if v.name == name:
                return v
        return None

    def save(self, path, params_path=None):
        with self.g.as_default():
            saver = tf.train.Saver(var_list=self.get_variables())
            save_path = saver.save(self.session, path)
        if params_path is not None:
            with open(params_path, 'w') as f:
                f.write(json.dumps(self.get_params()))
        return save_path

    def restore(self, path):
        with self.g.as_default():
            saver = tf.train.Saver(var_list=self.get_variables())
            saver.restore(self.session, path)

    def set_learning_rate(self, learning_rate):
        self.ensure_session()
        self.core.set_learning_rate(self.session, learning_rate)

    def get_learning_rate(self):
        self.ensure_session()
        return self.core.get_learning_rate(self.session)

    def get_params(self):
        return dict(
            core_arguments=self.core_factory.get_params(),
            log_dir=self.log_dir,
            session_config=self.session_config_dict,
            verbose=self.verbose,
            seed=self.seed,
        )

    @classmethod
    def create_from_files(cls, core_proc, params_path, vars_path):
        with open(params_path) as f:
            params = json.loads(f.read())
        core_arguments = params.pop('core_arguments', dict())
        cls.create_from_file(core_proc, params, core_arguments, vars_path)

    @classmethod
    def create_from_file(cls, core_proc, params, core_arguments, vars_path):
        class CoreFact(object):
            @staticmethod
            def create():
                return core_proc(core_arguments)

            @staticmethod
            def get_params():
                return core_arguments

        cnn = cls(CoreFact(), **params)
        cnn.ensure_session()
        cnn.restore(vars_path)
        return cnn

    def initialize_session(self):
        if self.need_logs:
            self.summary_writer = tf.summary.FileWriter(self.log_dir, self.g)
            if self.verbose > 0:
                full_log_path = os.path.abspath(self.log_dir)
                logging.info('Initialize logs, use: \ntensorboard --logdir=%s', full_log_path)
        self.session = tf.Session(config=self.session_config, graph=self.g)
        self._run(self.core.get_init_all_vars())

    def _run(self, ops, feed_dict=None):
        if self.debug_path:
            run_metadata = tf.RunMetadata()
            result = self.session.run(
                ops,
                feed_dict=feed_dict,
                options=tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE,
                    output_partition_graphs=True
                ),
                run_metadata=run_metadata
            )
            with open(self.debug_path, "a") as out:
                out.write(str(run_metadata))
            return result
        return self.session.run(ops, feed_dict=feed_dict)

    def ensure_session(self):
        if self.session is None:
            self.g.seed = self.seed
            with self.g.as_default():
                self.core.build_graph(self.g)
            self.initialize_session()

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        if self.g is not None:
            # self.g.close()
            self.g = None

    # todo it is similar to close so check and remove
    def destroy(self):
        """Terminates session and destroys graph."""
        self.session.close()
        self.g = None

    def reset_target_value(self):
        self.target_value = 0.0

    def get_target_value(self):
        return self.target_value

    def fit(self, x):
        if self.seed:
            np.random.seed(self.seed)
        self.ensure_session()

        self.target_value = 0.0
        fd = self.core.build_train_fd(x)
        ops_to_run = [self.core.get_minimizer()]
        loss = self.core.get_loss()
        if loss is not None:
            ops_to_run.append(loss)
        has_summary = False
        if self.need_logs:
            summary_op = self.core.get_summary_op()
            if summary_op is not None:
                has_summary = True
                ops_to_run.append(summary_op)
        result = self._run(ops_to_run, feed_dict=fd)
        # print('++++++++++++++++++++++++++++++')
        # wc = self._run(self.core.wc, feed_dict=fd)
        # print('wc', np.isnan(wc).sum())
        # input_c = self._run(self.core.vae.input_c, feed_dict=fd)
        # print('input_c', np.isnan(input_c).sum(), np.isneginf(input_c).sum(), np.isinf(input_c).sum())
        # q_psi = self.core.vae.q_psi
        # q_psi_arrs = self._run([q_psi.loc, q_psi.scale._diag], feed_dict=fd)
        # print([np.isnan(arr).sum() for arr in q_psi_arrs])
        # debug = self._run([self.core.vae._tau_psi.loc, self.core.vae._tau_psi.scale._diag], feed_dict=fd)
        # print('tau_psi inf', [np.isinf(arr).sum() for arr in debug])
        # print('tau_psi nan', [np.isnan(arr).sum() for arr in debug])
        # debug = self._run(self.core.vae.debug[0], feed_dict=fd)
        # print([(name, arr.shape) for name, arr in zip(self.core.vae.debug_names, debug)])
        # print([(name, np.isneginf(arr).sum()) for name, arr in zip(self.core.vae.debug_names, debug)])
        # print([(name, np.isinf(arr).sum()) for name, arr in zip(self.core.vae.debug_names, debug)])
        # print([(name, np.isnan(arr).sum()) for name, arr in zip(self.core.vae.debug_names, debug)])
        # debug = self._run(self.core.vae.debug[1], feed_dict=fd)
        # print([(name, np.isinf(arr).sum()) for name, arr in zip(self.core.vae.debug_names, debug)])
        # print([(name, np.isnan(arr).sum()) for name, arr in zip(self.core.vae.debug_names, debug)])
        # print(debug[0][0, :4], debug[1][0, :4], debug[2][0, :4], debug[3][0, :4])
        # log_tau_psi = debug[2]
        # nzt = np.nonzero(np.isinf(log_tau_psi))
        # nz0, nz1, nz2 = np.nonzero(np.isinf(log_tau_psi))
        # print(nz0.shape, nz1.shape, nz2.shape)
        # if nz0.shape[0] > 0:
        #     print(debug[3][nz0[0], nz1[0], nz2[0]])
        #     tau_psi = self.core.vae._tau_psi
        #     tau_psi_arrs = self._run([tau_psi.loc.shape, tau_psi.scale._diag.shape], feed_dict=fd)
        #     print(tau_psi_arrs[0].shape, tau_psi_arrs[1].shape)
        #     print(tau_psi_arrs[0][nz0[0], nz1[0], nz2[0]], tau_psi_arrs[1][nz0[0], nz1[0], nz2[0]])

        if has_summary:
            if loss is None:
                _, summary_str = result
                batch_target_value = None
            else:
                _, batch_target_value, summary_str = result
            self.summary_writer.add_summary(summary_str, self.steps)
            self.summary_writer.flush()
        else:
            if loss is None:
                _, = result
                batch_target_value = None
            else:
                _, batch_target_value = result
        if batch_target_value is not None:
            self.target_value += batch_target_value
        self.steps += 1

    def loss(self, x, is_train):
        fd = self.core.build_loss_fd(x, is_train)
        return self._run(self.core.get_loss(), feed_dict=fd)

    def loss_iter(self, x, is_train):
        loss = 0.0
        n = 0
        for bX in x:
            fd = self.core.build_loss_fd(bX, is_train)
            loss += self._run(self.core.get_loss(), feed_dict=fd)
            n += 1
        return loss / n

    def predict(self, x):
        self.ensure_session()
        fd = self.core.build_predict_fd(x)
        return self._run(self.core.get_predict(), feed_dict=fd)

    def predict_iter(self, x, custom_proc=None, tensor_cnt=1):
        self.ensure_session()
        if tensor_cnt == 1:
            output = []
        else:
            output = [[] for _ in range(tensor_cnt)]
        for bX in x:
            fd = self.core.build_predict_fd(bX)
            pred = self._run(self.core.get_predict(), feed_dict=fd)
            # print(tensor_cnt, type(pred), [t.shape for t in pred])
            if custom_proc is not None:
                pred = custom_proc(bX, pred)
            if tensor_cnt == 1:
                output.append(pred)
            else:
                for l, t in zip(output, pred):
                    # print(len(l), t.shape)
                    l.append(t)
        if tensor_cnt == 1:
            return np.concatenate(output, axis=0)
        # print([[t.shape for t in l] for l in output])
        return [np.concatenate(l, axis=0) for l in output]

    def get_vars(self, x, var=None):
        self.ensure_session()
        fd = self.core.build_predict_fd(x)
        return self._run(self.core.test if var is None else var, feed_dict=fd)


def load_model(model_path, params_or_path, core_from_params_proc, custom_core_params=None):
    if custom_core_params is None:
        custom_core_params = dict()
    params = read_json(params_or_path) if isinstance(params_or_path, str) else params_or_path
    core_arguments = params.pop('core_arguments', dict())
    core_arguments.update(custom_core_params)
    return TfModel.create_from_file(core_from_params_proc, params, core_arguments, model_path)


def load_std_model(model_path, core_from_params_proc, **custom_core_params):
    return load_model(
        os.path.join(model_path, 'model.bin'),
        os.path.join(model_path, 'model_params.txt'),
        core_from_params_proc,
        custom_core_params=custom_core_params
    )
