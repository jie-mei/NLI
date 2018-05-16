import typing as t

import tensorflow as tf


def linear(
        inputs: tf.Tensor,
        dim: int = -1,
        activation_fn: t.Callable = tf.nn.relu,
        weight_init: float = None,
        bias_init: float = 0,
        bias: bool = True,
        keep_prob: float = 1.0,
        scope: t.Union[str, tf.VariableScope] = None,
        reuse: bool = tf.AUTO_REUSE,
        device: str = None
        ):
    """
    Inputs:  3D-Tensor [batch, seq_len, input_dim], or
             2D-Tensor [batch, input_dim]
    Returns: 3D-Tensor [batch, seq_len, dim], or
             2D-Tensor [batch, dim]
    """
    dim = dim if dim > 0 else int(inputs.shape[-1])
    t = tf.nn.dropout(inputs, keep_prob)
    with tf.device(device):
        with tf.variable_scope(scope if scope else 'linear', reuse=reuse):
            t_shape = tf.shape(t)
            w = tf.get_variable('weight',
                    shape=[t.get_shape()[-1], dim],
                    dtype=tf.float32,
                    initializer=weight_init)
            output_rank = len(t.get_shape())
            if output_rank == 3:
                t = tf.reshape(t, [-1, t.shape[2]])
            t = tf.matmul(t, w)
            if bias:
                if bias_init == 0:
                    init = tf.initializers.zeros()
                else:
                    init = tf.initializers.constant(bias_init)
                b = tf.get_variable('bias',
                        shape=[dim],
                        dtype=tf.float32,
                        initializer=init)
                t += b
            if output_rank == 3:
                t = tf.reshape(t, [-1, t_shape[1], dim])
            t = activation_fn(t) if activation_fn else t
    return t
