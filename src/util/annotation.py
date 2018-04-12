""" Customized annotations. """

import tensorflow as tf


def name_scope(name):
    """ Annotation for wrapping a function into namespace. """
    def func_decoder(func):
        def wrapped_func(*args, **kwargs):
            with tf.name_scope(name):
                return func(*args, **kwargs)
        return wrapped_func
    return func_decoder


def variable_scope(name):
    """ Annotation for wrapping a function into namespace. """
    def func_decoder(func):
        def wrapped_func(*args, **kwargs):
            with tf.variable_scope(name):
                return func(*args, **kwargs)
        return wrapped_func
    return func_decoder


def print_section(func):
    """ Annotation for printing segmentation lines in console. """
    def wrapped_func(*args, **kwargs):
        print("=" * 80)
        func_out = func(*args, **kwargs)
        print("=" * 80)
        print()
        return func_out
    return wrapped_func
