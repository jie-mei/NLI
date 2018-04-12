import tensorflow as tf

from typing import Union, Callable

def word_conv(
        inputs: tf.Tensor,
        dim: int = -1,
        activation_fn: Callable = tf.nn.relu,
        weight_stddev: float = 0.01,
        l2_scale: float = 0.005,
        bias_init: float = 0,
        scope: Union[str, tf.VariableScope] = None,
        reuse: bool = False,
        name: str = 'fully_connected',
        ):
    pass
    """
    Inputs:
        3-D Tensor [batch, seq_len, input_dim]

    Returns:
        3-D Tensor [batch, seq_len, dim]
    """
    dim  = dim if dim > 0 else int(inputs.shape[-1])
    with tf.variable_scope(scope, default_name=name) as scope:
        t = tf.contrib.layers.fully_connected(inputs, dim,
                activation_fn=None,
                weights_initializer=tf.initializers.truncated_normal(
                        stddev=weight_stddev),
                weights_regularizer=tf.contrib.layers.l2_regularizer(
                        scale=l2_scale),
                biases_initializer=tf.constant_initializer(bias_init),
                scope=scope,
                reuse=reuse)
        t = activation_fn(t)
        return t
