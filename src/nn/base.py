""" Base NN model. """

from abc import ABC

import tensorflow as tf
import numpy as np

import data
from util.display import ReprMixin
from util.log import exec_log as log


WORD_SEQ_LEN = 16


class Model(ReprMixin, ABC):
    """ A base NN model to conduct pairwised text analysis. """

    def __init__(self):
        with tf.name_scope('input'):
            self.handle = tf.placeholder(tf.string, shape=[], name='handle')
            self.data_iterator = tf.data.Iterator.from_string_handle(
                    string_handle=self.handle,
                    output_types=(tf.int32,) * 11,
                    output_shapes=(tf.TensorShape([None, None]),
                                   tf.TensorShape([None, None]),
                                   tf.TensorShape([None]),
                                   tf.TensorShape([None]),
                                   tf.TensorShape([None]),
                                   tf.TensorShape([None, None, WORD_SEQ_LEN]),
                                   tf.TensorShape([None, None, WORD_SEQ_LEN]),
                                   tf.TensorShape([None, None, 4]),
                                   tf.TensorShape([None, None, 4]),
                                   tf.TensorShape([None, None]),
                                   tf.TensorShape([None, None])))
            iter_next = self.data_iterator.get_next()
        self.x1    = tf.identity(iter_next[0],  name='id1')
        self.x2    = tf.identity(iter_next[1],  name='id2')
        self.y     = tf.identity(iter_next[2],  name='y')
        self.len1  = tf.identity(iter_next[3],  name='len1')
        self.len2  = tf.identity(iter_next[4],  name='len2')
        self.char1 = tf.identity(iter_next[5],  name='char1')
        self.char2 = tf.identity(iter_next[6],  name='char2')
        self.temp1 = tf.identity(iter_next[7],  name='temp1')
        self.temp2 = tf.identity(iter_next[8],  name='temp2')
        self.tag1  = tf.identity(iter_next[9],  name='tag1')
        self.tag2  = tf.identity(iter_next[10], name='tag2')

    def count_parameters(self):
        total = 0
        for var in tf.trainable_variables():
            num = 1
            for dim in var.get_shape():
                num *= int(dim)
            total += num
        return total


class SoftmaxCrossEntropyMixin(ABC):

    def evaluate_and_loss(self, y_hat):
        with tf.name_scope('predict'):
            probs = tf.nn.softmax(y_hat)
            self.prediction = tf.argmax(probs, axis=1, output_type=tf.int32)
        with tf.name_scope('accuracy'):
            self.performance = tf.reduce_mean(
                    tf.cast(tf.equal(self.prediction, self.y), tf.float32))
        with tf.name_scope('loss'):
            #labels = tf.one_hot(self.y, self._class_num, dtype=tf.float32)
            ce_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=y_hat, labels=self.y))
            rglz_loss = tf.reduce_sum(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = tf.add(ce_loss, rglz_loss)
