from typing import Callable, Union

import tensorflow as tf

import op

def forward_product(
        x1: tf.Tensor,
        x2: tf.Tensor,
        activation_fn: Callable = tf.nn.relu,
        scope: Union[str, tf.VariableScope] = None,
        **kwargs,
        ):
    """
    Inputs:
        3-D Tensor [batch, seq_len_1, embed_dim]
        3-D Tensor [batch, seq_len_2, embed_dim]

    Returns:
        3-D Tensor [batch, seq_len_1, seq_len_2]
    """
    scope = scope if scope else 'forward_production'
    forward = lambda x: op.forward(x, scope=scope, **kwargs)
    sim = tf.expand_dims(forward(x1), 2) * tf.expand_dims(forward(x2), 1)
    sim = tf.reduce_sum(sim, axis=3)
    return sim
