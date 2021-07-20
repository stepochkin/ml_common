import os

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops

from stepin.ml.tf_model import load_std_model


def tf_disable_output():
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def tf_mat_repeat(arr, n):
    width = tf.shape(arr)[1]
    arr = tf.tile(arr, [1, n])
    return tf.reshape(arr, [-1, width])


def tf_vec_repeat(arr, n):
    arr = tf.reshape(arr, [-1, 1])
    arr = tf.tile(arr, [1, n])
    return tf.reshape(arr, [-1])


def tf_sparse_uop(t, uop):
    return tf.SparseTensor(t.indices, uop(t.values), t.dense_shape)


def repeat_range(counts, dtype=dtypes.int64, name=None):
    """Repeat integers given by range(len(counts)) each the given number of times.
    Example behavior:
    [0, 1, 2, 3] -> [1, 2, 2, 3, 3, 3]
    Args:
        counts: 1D tensor with dtype=int32.
        dtype: result dtype
        name: optional name for operation.
    Returns:
        1D tensor with dtype=int32 and dynamic length giving the repeated integers.
    """
    with ops.name_scope(name, 'repeat_range', [counts]) as scope:
        counts = ops.convert_to_tensor(counts, name='counts')

        def cond(_, i):
            return i < size

        def body(output, i):
            value = tf.fill(counts[i:i+1], tf.cast(i, dtype=dtype))
            return output.write(i, value), i + 1

        size = array_ops.shape(counts)[0]
        init_output_array = tensor_array_ops.TensorArray(
            dtype=dtype, size=size, infer_shape=False
        )
        output_array, num_writes = control_flow_ops.while_loop(
            cond, body, loop_vars=[init_output_array, 0]
        )

        return control_flow_ops.cond(
            num_writes > 0,
            output_array.concat,
            lambda: array_ops.zeros(shape=[0], dtype=dtype),
            name=scope
        )


def start_ranges(counts, name=None):
    with ops.name_scope(name, 'start_ranges', [counts]) as scope:
        counts = ops.convert_to_tensor(counts, name='counts')

        def cond(_, i):
            return i < size

        def body(output, i):
            value = tf.range(counts[i])
            return output.write(i, value), i + 1

        size = array_ops.shape(counts)[0]
        init_output_array = tensor_array_ops.TensorArray(
            dtype=dtypes.int32, size=size, infer_shape=False
        )
        output_array, num_writes = control_flow_ops.while_loop(
            cond, body, loop_vars=[init_output_array, 0]
        )

        return control_flow_ops.cond(
            num_writes > 0,
            output_array.concat,
            lambda: array_ops.zeros(shape=[0], dtype=dtypes.int32),
            name=scope
        )


def start_ranges_k(counts, k, name=None):
    with ops.name_scope(name, 'repeat_range_k', [counts]):
        counts = ops.convert_to_tensor(counts, name='counts')
        ranges = start_ranges(counts)
        return tf_vec_repeat(ranges, k)


class TfFullConnect(object):
    def __init__(
        self,
        input_size, l2_loss, hidden_sizes, name, is_last=True,
        is_first_sparse=False, is_trainable=True, init_values=None, l1_loss=None,
        input_size_ext=None, dtype=None
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.name = name
        self.is_last = is_last
        self.is_first_sparse = is_first_sparse
        self.is_trainable = is_trainable
        self.init_values = init_values
        self.input_size_ext = input_size_ext
        self.dtype = dtype

        self.w = []
        self.b = []
        self.ext_w = None

        psize = input_size
        for hsize_i, hsize in enumerate(self.hidden_sizes):
            with tf.variable_scope(name + str(hsize_i)):
                if init_values is None:
                    W = tf.get_variable(
                        "W", shape=[psize, hsize],
                        initializer=(tf.contrib.layers.xavier_initializer()),
                        dtype=dtype,
                        trainable=is_trainable
                    )
                    b = tf.get_variable(
                        "b", shape=[hsize],
                        initializer=(tf.constant_initializer(0.1)),
                        dtype=dtype,
                        trainable=is_trainable
                    )
                else:
                    W = tf.get_variable(
                        "W",
                        # shape=[psize, hsize],
                        initializer=init_values.getw(hsize_i),
                        dtype=dtype,
                        trainable=is_trainable
                    )
                    b = tf.get_variable(
                        "b",
                        # shape=[hsize],
                        initializer=(init_values.getb(hsize_i)),
                        dtype=dtype,
                        trainable=is_trainable
                    )
                self.w.append(W)
                self.b.append(b)
                if l2_loss is not None:
                    l2_loss += tf.nn.l2_loss(W)
                    # l2_loss += tf.nn.l2_loss(b)
                if l1_loss is not None:
                    reg = tf.contrib.layers.l1_regularizer(scale=1.0)
                    l1_loss += tf.contrib.layers.apply_regularization(reg, [W])

                if (hsize_i == 0) and (input_size_ext is not None):
                    if init_values is None:
                        W = tf.get_variable(
                            "W_ext", shape=[input_size_ext, hsize],
                            initializer=(tf.contrib.layers.xavier_initializer()),
                            dtype=dtype,
                            trainable=is_trainable
                        )
                    else:
                        W = tf.get_variable(
                            "W_ext",
                            initializer=init_values.getw_ext(hsize_i),
                            dtype=dtype,
                            trainable=is_trainable
                        )
                    self.ext_w = W
                    if l2_loss is not None:
                        l2_loss += tf.nn.l2_loss(W)
                    if l1_loss is not None:
                        reg = tf.contrib.layers.l1_regularizer(scale=1.0)
                        l1_loss += tf.contrib.layers.apply_regularization(reg, [W])
            psize = hsize
        self.l2_loss = l2_loss
        self.l1_loss = l1_loss

    def build(self, input_data, name, input_data_ext=None, normalizer_fn=None):
        layer = input_data
        for hsize_i, _ in enumerate(self.hidden_sizes):
            with tf.variable_scope(self.name + '_' + name + str(hsize_i)):
                W = self.w[hsize_i]
                b = self.b[hsize_i]
                if self.is_first_sparse and (hsize_i == 0):
                    layer = tf.sparse_tensor_dense_matmul(layer, W) + b
                else:
                    layer = tf.nn.xw_plus_b(layer, W, b, name="xw_plus_b")

                if (hsize_i == 0) and (self.ext_w is not None):
                    layer += tf.matmul(input_data_ext, self.ext_w, name="xw_ext")

                if normalizer_fn is not None:
                    layer = normalizer_fn(layer)

                if not self.is_last or (hsize_i < len(self.hidden_sizes) - 1):
                    layer = tf.nn.relu(layer, name="relu")
        return layer


def tf_build_full_connect(
    input_data, input_size, l2_loss, hidden_sizes, name, is_last,
    is_first_sparse=False, is_trainable=True, init_values=None, l1_loss=None,
    input_data_ext=None, input_size_ext=None, dtype=None
):
    layer = input_data
    is_layer_list = isinstance(layer, (list, tuple))
    if not is_layer_list:
        layer = [layer]
    is_l2_loss_list = isinstance(l2_loss, (list, tuple))
    if not is_l2_loss_list:
        l2_loss = [l2_loss]
    psize = input_size
    for hsize_i, hsize in enumerate(hidden_sizes):
        # logging.debug('hsize %s', hsize)
        with tf.variable_scope(name + str(hsize_i)):
            if init_values is None:
                W = tf.get_variable(
                    "W", shape=[psize, hsize],
                    initializer=(tf.contrib.layers.xavier_initializer()),
                    dtype=dtype,
                    trainable=is_trainable
                )
                b = tf.get_variable(
                    "b", shape=[hsize],
                    initializer=(tf.constant_initializer(0.1)),
                    dtype=dtype,
                    trainable=is_trainable
                )
            else:
                W = tf.get_variable(
                    "W",
                    # shape=[psize, hsize],
                    initializer=init_values.getw(hsize_i),
                    dtype=dtype,
                    trainable=is_trainable
                )
                b = tf.get_variable(
                    "b",
                    # shape=[hsize],
                    initializer=(init_values.getb(hsize_i)),
                    dtype=dtype,
                    trainable=is_trainable
                )

            if l2_loss is not None:
                w_loss = tf.nn.l2_loss(W)
                l2_loss = [l + w_loss for l in l2_loss]
                # l2_loss += tf.nn.l2_loss(b)
            if l1_loss is not None:
                reg = tf.contrib.layers.l1_regularizer(scale=1.0)
                w_loss = tf.contrib.layers.apply_regularization(reg, [W])
                l1_loss = [l + w_loss for l in l1_loss]
            if is_first_sparse and (hsize_i == 0):
                layer = [tf.sparse_tensor_dense_matmul(l, W) + b for l in layer]
            else:
                layer = [tf.nn.xw_plus_b(l, W, b, name="xw_plus_b") for l in layer]

            if (hsize_i == 0) and (input_data_ext is not None):
                if init_values is None:
                    W = tf.get_variable(
                        "W_ext", shape=[input_size_ext, hsize],
                        initializer=(tf.contrib.layers.xavier_initializer()),
                        dtype=dtype,
                        trainable=is_trainable
                    )
                else:
                    W = tf.get_variable(
                        "W_ext",
                        initializer=init_values.getw_ext(hsize_i),
                        dtype=dtype,
                        trainable=is_trainable
                    )

                if l2_loss is not None:
                    w_loss = tf.nn.l2_loss(W)
                    l2_loss = [l + w_loss for l in l2_loss]
                if l1_loss is not None:
                    reg = tf.contrib.layers.l1_regularizer(scale=1.0)
                    w_loss = tf.contrib.layers.apply_regularization(reg, [W])
                    l1_loss = [l + w_loss for l in l1_loss]
                ext = tf.matmul(input_data_ext, W, name="xw_ext")
                for l_i in range(len(layer)):
                    layer[l_i] = layer[l_i] + ext

            if not is_last or (hsize_i < len(hidden_sizes) - 1):
                layer = [tf.nn.relu(l, name="relu") for l in layer]

        psize = hsize

    if not is_layer_list:
        layer = layer[0]
    if not is_l2_loss_list:
        l2_loss = l2_loss[0]
    if l1_loss is None:
        return layer, l2_loss
    else:
        return layer, l2_loss, l1_loss


class FullConnectInitValues(object):
    def __init__(self, model_path, core_from_params, name_prefix, **custom_core_params):
        mlnet = load_std_model(model_path, core_from_params, **custom_core_params)
        self.variables = mlnet.get_variable_dict(regexp='^hidden_.*')
        del mlnet
        self.name_prefix = name_prefix

    def getw(self, i):
        return self.variables[self.name_prefix + str(i) + '/W:0']

    def getb(self, i):
        return self.variables[self.name_prefix + str(i) + '/b:0']
