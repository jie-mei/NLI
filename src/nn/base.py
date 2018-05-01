""" Base NN model. """

from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf
import numpy as np

import data
from util.display import ReprMixin
from util.log import exec_log as log


class Model(ReprMixin, ABC):
    """ A base NN model to conduct pairwised text analysis. """

    def __init__(self):
        self.data_iterator = tf.data.Iterator.from_structure(
                output_types=(tf.int32,) * 5,
                output_shapes=(tf.TensorShape([None, None]),
                               tf.TensorShape([None, None]),
                               tf.TensorShape([None]),
                               tf.TensorShape([None]),
                               tf.TensorShape([None])))
        self.x1, self.x2, self.y, self.len1, self.len2 = \
                self.data_iterator.get_next()

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
