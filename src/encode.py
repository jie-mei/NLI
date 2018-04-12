from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf

import nn
from util.annotation import name_scope
from util.layers import lstm
from util.display import ReprMixin


class Encoder(ReprMixin, ABC):
    @abstractmethod
    def encode(self, x1: tf.Tensor, x2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass


class NoEncode(Encoder):

    def __init__(self, **kwargs) -> None:
        pass

    def encode(self, x1: tf.Tensor, x2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return x1, x2


class CNN(Encoder):

    def __init__(self,
                 output_dim: int = 50,
                 l2_scale: float = 0.005,
                 weight_stddev: float = 0.05,
                 bias_init: float = 1e-04,
                 **kwargs
                 ) -> None:
        self.output_dim = output_dim
        self.l2_scale = l2_scale
        self.weight_stddev = weight_stddev
        self.bias_init = bias_init

    def encode(self, x1: tf.Tensor, x2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Construct a word-based convolution for learning task-specific
        # representation
        with tf.variable_scope('encode') as scope:
            def to_learnable_repr(t):
                t = tf.expand_dims(t, -1)
                t = tf.contrib.layers.conv2d(
                        inputs=t,
                        num_outputs=self.output_dim,
                        kernel_size=(int(t.shape[1]), 1),
                        stride=1,
                        padding='VALID',
                        activation_fn=None,
                        weights_initializer=tf.initializers.truncated_normal(
                                stddev=self.weight_stddev),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(
                                scale=self.l2_scale),
                        biases_initializer=tf.constant_initializer(
                                self.bias_init),
                        scope=scope,
                        reuse=tf.AUTO_REUSE)
                t = tf.nn.tanh(t)
                t = tf.transpose(t, [0, 3, 2, 1])
                t = tf.squeeze(t, [3])
                return t
            x1, x2 = map(to_learnable_repr, (x1, x2))
            return x1, x2


class RNN(Encoder):

    def __init__(self,
                 model: 'nn.Model',
                 output_dim: int = 200,
                 cell_type: str = 'gru',
                 ) -> None:
        self.cell_type = cell_type
        self.output_dim = output_dim
        self.x1_ids = model.x1
        self.x2_ids = model.x2

    def encode(self, x1: tf.Tensor, x2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Construct a word-based convolution for learning task-specific
        # representation
        with tf.variable_scope('encode', reuse=tf.AUTO_REUSE):
            def to_learnable_repr(x_embeds, x_ids):
                seq_len = tf.count_nonzero(x_ids, axis=1, dtype=tf.float32)
                t = tf.transpose(x_embeds, [0, 2, 1])
                t = lstm(t, seq_len,
                        hidden_size=self.output_dim,
                        cell_type='gru')
                t = tf.transpose(t, [0, 2, 1])
                return t
            return (to_learnable_repr(x1, self.x1_ids),
                    to_learnable_repr(x2, self.x2_ids))
