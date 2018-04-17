from functools import reduce

import tensorflow as tf


def print_trainable_variables():
    fmt = '%-40s  %-20s  %10d'
    total = 0
    for v in tf.trainable_variables():
        num = reduce(lambda x, y: (x)*y, map(int, v.get_shape()))
        total += num
        print(fmt % (v.name, v.get_shape(), num))
    print(fmt % ('total', '', total))
