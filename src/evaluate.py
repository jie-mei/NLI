from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf

import nn
from util.annotation import name_scope
from util.display import ReprMixin


class Evaluator(ReprMixin, ABC):

    @abstractmethod
    def loss(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def predict(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def evaluate(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        pass


class SoftmaxCE(Evaluator):

    def __init__(self,
                 model: 'nn.Model',
                 l2_scale: float = 0.001,
                 weight_stddev: float = 0.05,
                 bias_stddev: float = 1e-04,
                 ) -> None:
        self.l2_scale = l2_scale
        self.weight_stddev = weight_stddev
        self.bias_stddev = bias_stddev
        self.gt = model.y

    def _fully_connected(self, x1: tf.Tensor, x2: tf.Tensor):
        if not hasattr(self, '_fc'):
            t = tf.concat([x1, x2], 1)
            with tf.variable_scope('fully_connected') as scope:
                t = tf.contrib.layers.fully_connected(
                        inputs=t,
                        num_outputs=2,
                        activation_fn=None,
                        weights_initializer=tf.initializers.truncated_normal(
                                stddev=self.weight_stddev),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(
                                scale=self.l2_scale),
                        biases_initializer=tf.constant_initializer(
                                self.bias_stddev),
                        scope=scope,
                        reuse=tf.AUTO_REUSE)
            self._fc = tf.nn.sigmoid(t)
        return self._fc

    def loss(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        fc = self._fully_connected(x1, x2)
        with tf.name_scope('loss'):
            ce_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=fc,
                            labels=self.gt))  # type: ignore
            l2_loss = tf.reduce_sum(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES))
            return tf.add(ce_loss, l2_loss)

    def predict(self, x1: tf.Tensor, x2: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        probs = tf.contrib.layers.softmax(self._fully_connected(x1, x2))
        return tf.cast(probs[:, 1] > 0.5, tf.int32), probs[:, 1]

    def evaluate(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('accuracy'):
            return tf.reduce_mean(tf.cast(tf.equal(pred, gt), tf.float32))


class CosineMSE(Evaluator):

    def __init__(self,
                 model: 'nn.Model',
                 l2_scale: float = 0.001,
                 weight_stddev: float = 0.05,
                 bias_stddev: float = 1e-04,
                 ) -> None:
        self.l2_scale = l2_scale
        self.weight_stddev = weight_stddev
        self.bias_stddev = bias_stddev
        self.gt = model.y

    def _cosine_similarity(self, x1: tf.Tensor, x2: tf.Tensor):
        if not hasattr(self, '_cos'):
            x1, x2 = tf.nn.l2_normalize(x1, 1), tf.nn.l2_normalize(x2, 1)
            self._cos = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
        return self._cos

    def loss(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        sim = self._cosine_similarity(x1, x2)
        #threshold = tf.Variable(
        #        name='threshold',
        #        initial_value=tf.constant(0.5, shape=[1], dtype=tf.float32))
        with tf.name_scope('loss'):
            mse_loss = tf.losses.mean_squared_error(
                    tf.cast(self.gt, tf.float32), sim)
            l2_loss = tf.reduce_sum(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES))
            return tf.add(mse_loss, l2_loss)

    def predict(self, x1: tf.Tensor, x2: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        sim = self._cosine_similarity(x1, x2)
        return tf.cast(sim > 0.5, tf.int32), sim

    def evaluate(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('accuracy'):
            return tf.reduce_mean(tf.cast(tf.equal(pred, gt), tf.float32))


class TrainableCosineMSE(Evaluator):

    def __init__(self,
                 model: 'nn.Model',
                 l2_scale: float = 0.001,
                 weight_stddev: float = 0.05,
                 bias_stddev: float = 1e-04,
                 ) -> None:
        self.l2_scale = l2_scale
        self.weight_stddev = weight_stddev
        self.bias_stddev = bias_stddev
        self.gt = model.y

    def _cosine_similarity(self, x1: tf.Tensor, x2: tf.Tensor):
        if not hasattr(self, '_cos'):
            x1, x2 = tf.nn.l2_normalize(x1, 1), tf.nn.l2_normalize(x2, 1)
            self.factor = tf.Variable(
                    name='factor',
                    initial_value=tf.constant(1.0, shape=[1], dtype=tf.float32))
            self._cos = tf.reduce_sum(tf.multiply(x1, x2), axis=1) * self.factor
        return self._cos

    def loss(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        sim = self._cosine_similarity(x1, x2)
        with tf.name_scope('loss'):
            mse_loss = tf.losses.mean_squared_error(
                    tf.cast(self.gt, tf.float32), sim)
            l2_loss = tf.reduce_sum(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES))
            return tf.add(mse_loss, l2_loss)

    def predict(self, x1: tf.Tensor, x2: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        sim = self._cosine_similarity(x1, x2)
        return tf.cast(sim > 0.5, tf.int32), sim

    def evaluate(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('accuracy'):
            return tf.reduce_mean(tf.cast(tf.equal(pred, gt), tf.float32))
