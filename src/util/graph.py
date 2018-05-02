from functools import reduce

import tensorflow as tf


def print_trainable_variables():
    output = ''
    fmt = '%-40s  %-20s  %10d'
    total = 0
    for v in tf.trainable_variables():
        num = reduce(lambda x, y: (x)*y, map(int, v.get_shape()))
        total += num
        output += fmt % (v.name, v.get_shape(), num) + '\n'
    output += fmt % ('total', '', total) + '\n'
    return output
