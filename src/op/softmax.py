import tensorflow as tf

from typing import Union

def softmax(
        logits: tf.Tensor,
        axis: int = -1,
        scope: Union[str, tf.VariableScope] = None,
        ):
    dims = len(logits.shape)
    axis %= dims
    if dims - axis > 1:
        perm = list(range(dims))
        perm[-1], perm[axis] = perm[axis], perm[-1] 
        logits = tf.transpose(logits, perm=perm)
    logits = tf.contrib.layers.softmax(logits, scope=scope)
    if dims - axis > 1:
        logits = tf.transpose(logits, perm=perm)
    return logits
    

