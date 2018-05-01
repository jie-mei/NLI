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

    def count_parameters(self):
        total = 0
        for var in tf.trainable_variables():
            num = 1
            for dim in var.get_shape():
                num *= int(dim)
            total += num
        return total


    def feed(self,
            dataset: data.Dataset,
            batch_size: int = 64,
            shuffle_buffer_size: int = 40960,
            prefetch_buffer_size: int = -1,
            repeat_num: int = 1,
            )-> None:
        """ Fit a dataset as an input pipeline for batched evaluation.

        When feeding into the model, input sequences are padded to the same
        length. The dataset is shuffled if the `shuffle_buffer_size` is greater
        than 1.

        Args:
            dataset: A dataset.
            batch_size: The batch size.
            shuffle_buffer_size: The buffer size for random shuffling. Disable
                random shuffling when `shuffle_buffer_size` is smaller than or
                equal to 1.
            prefetch_buffer_size: The buffer size for prefetching. This
                parameter is used, when shuffling is permitted and tensorflow
                version is lower than 1.6. When given a non-positive value,
                `prefetch_buffer_size` will adapt to 64 * `batch_size`.
            repeat_time: The number of times the records in the dataset are
                repeated.
        """
        padded_shapes = ([None], [None], [])  # type: tuple
        dset = tf.data.Dataset.from_generator(
                lambda: ((x1, x2, y, len(x1), len(x2))
                         for x1, x2, y in
                         zip(dataset.x1_ids, dataset.x2_ids, dataset.labels)),
                output_types=(tf.int32,) * 5,
                output_shapes=(tf.TensorShape([None]),
                               tf.TensorShape([None]),
                               tf.TensorShape([]),
                               tf.TensorShape([]),
                               tf.TensorShape([])))

        if shuffle_buffer_size > 1:
            if tf.__version__ >= '1.6':
                # Recommended by the Tensorflow input pipeline performance guide.
                dset = dset.apply(tf.contrib.data.shuffle_and_repeat(
                    shuffle_buffer_size, repeat_num))
            else:
                dset = dset.shuffle(shuffle_buffer_size).repeat(repeat_num)
        else:
            dset = dset.repeat(repeat_num)

        # Pack records with similar lengthes as batch.
        if tf.__version__ >= '1.8':
            log.debug('Generate batches using'
                      'tf.contrib.data.bucket_by_sequence_length')
            dset = dset.apply(tf.contrib.data.bucket_by_sequence_length(
                    lambda x1, x2, y, len1, len2: tf.maximum(len1, len2),
                    [20, 50],
                    [batch_size] * 3))
        else:
            log.debug('Generate batches using tf.contrib.data.group_by_window')
            def bucketing(x1, x2, y, len1, len2):
                size = tf.maximum(len1, len2)
                bucket = tf.case([(size < 20, lambda: 1),
                                  (size > 50, lambda: 2)],
                                 default=lambda: 0,
                                 exclusive=True)
                return tf.to_int64(bucket)
            dset = dset.apply(tf.contrib.data.group_by_window(
                    key_func=bucketing,
                    reduce_func=lambda _, data: data.padded_batch(batch_size,
                            padded_shapes = ([None], [None], [], [], [])),
                    window_size=batch_size))

        if prefetch_buffer_size <= 0:
            prefetch_buffer_size = 64 * batch_size
        dset = dset.prefetch(buffer_size=prefetch_buffer_size)

        iterator = tf.data.Iterator.from_structure(
                dset.output_types, dset.output_shapes)
        self.init = iterator.make_initializer(dset)
        self.x1, self.x2, self.y, self.len1, self.len2 = iterator.get_next()


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
