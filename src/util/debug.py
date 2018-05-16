import tensorflow as tf


def tf_Print(input_, data, **kwargs):
    kwargs = {'summarize': 2**30, 'first_n': 1, **kwargs}
    return tf.Print(input_, data, **kwargs)
