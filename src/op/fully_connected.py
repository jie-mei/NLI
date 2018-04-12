import tensorflow as tf

from typing import Union, Callable

def fc(
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
