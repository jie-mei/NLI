import tensorflow as tf

from typing import Union, Callable


def _get_variable(name, shape, stddiv, mean=0, dtype=tf.float32):
    """ Get variable with the default setup. """
    if stddiv:
        init = tf.initializers.truncated_normal(mean, stddiv, dtype=dtype)
    else:
        init = tf.initializers.constant(mean, dtype=dtype)
    return tf.get_variable(name, shape, initializer=init, dtype=dtype)


def _forward_impl1(
        inputs: tf.Tensor,
        dim: int = -1,
        activation_fn: Callable = None,
        keep_prob: float = 1.0,
        weight_stddev: float = 0.01,
        scope: Union[str, tf.VariableScope] = None,
        reuse: bool = False,
        ):
    """ Light-weighted feed-forward without regularization.

    Inputs:
        n-D Tensor [..., input_dim]

    Returns:
        n-D Tensor [..., dim]
    """
    scope = scope if scope else 'forward'
    dim = dim if dim > 0 else int(inputs.shape[-1])
    with tf.variable_scope(scope, reuse=reuse):
        w = _get_variable('weight', [int(inputs.shape[-1]), dim], weight_stddev)
        b = _get_variable('bias', [dim], 0)
    # Batch matrix multiplication.
    shape = list(inputs.shape)
    t = tf.reshape(inputs, [-1, shape[-1]])
    t = tf.matmul(t, w)
    t = tf.reshape(t, [-1] + shape[1:-1] + [dim])
    #while len(w.shape) < len(inputs.shape): w = tf.expand_dims(w, 0)
    #w *= tf.ones(inputs.get_shape(), dtype=tf.float32)  # broadcast to shape
    while len(b.shape) < len(inputs.shape): b = tf.expand_dims(b, 0)
    #t = tf.matmul(inputs, w) + b
    t = t + b
    t = tf.nn.dropout(t, keep_prob)
    t = activation_fn(t) if activation_fn else t
    return t


def _forward_impl2(
        inputs: tf.Tensor,
        dim: int = -1,
        activation_fn: Callable = tf.nn.relu,
        keep_prob: float = 1.0,
        weight_stddev: float = 0.01,
        bias_init: float = 0,
        rglz_type: str = None,
        rglz_scale: float = 0.01,
        scope: Union[str, tf.VariableScope] = None,
        reuse: bool = False,
        ):
    """
    Inputs:
        3-D Tensor [batch, seq_len, input_dim]

    Returns:
        3-D Tensor [batch, seq_len, dim]
    """
    scope = scope if scope else 'fully_connected'
    dim = dim if dim > 0 else int(inputs.shape[-1])
    rglz = {'l1': tf.contrib.layers.l1_regularizer(rglz_scale),
            'l2': tf.contrib.layers.l2_regularizer(rglz_scale),
            }.get(rglz_type, None)
    t = tf.contrib.layers.fully_connected(inputs, dim,
            activation_fn=None,
            weights_initializer=tf.initializers.truncated_normal(
                    stddev=weight_stddev),
            weights_regularizer=rglz,
            biases_initializer=tf.constant_initializer(bias_init),
            scope=scope,
            reuse=reuse)
    t = tf.nn.dropout(t, keep_prob)
    t = activation_fn(t) if activation_fn else t
    return t


def forward(inputs, **kwargs):
    return _forward_impl2(inputs, **kwargs)
