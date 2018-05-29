from functools import reduce

import tensorflow as tf


def print_trainable_variables():
    output = ''
    fmt = '%-50s  %-20s  %10d'
    total = 0
    for v in tf.trainable_variables():
        try:
            num = reduce(lambda x, y: (x)*y, map(int, v.get_shape()))
            total += num
            output += fmt % (v.name, v.get_shape(), num) + '\n'
        except ValueError:
            output += fmt % (v.name, 'Unknown', 0) + '\n'
    output += fmt % ('total', '', total) + '\n'
    return output
