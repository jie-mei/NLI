from abc import ABC, abstractmethod
import typing as t

import numpy as np
import tensorflow as tf

import embed
from nn.decomposable import Decomposable


class Ngram(Decomposable, ABC):

    def __init__(self,
            ngram_size: t.Union[int, t.List[int]] = [2, 3],
            **kwargs
            )-> None:
        self.ngram_size = ngram_size \
                if isinstance(ngram_size, list) else [ngram_size]
        super(Ngram, self).__init__(**kwargs)

    def post_project(self, x: tf.Tensor):
        for size in self.ngram_size:
            x = tf.concat([x, self.ngram_embed(x, size)], 1)
        return x

    @abstractmethod
    def ngram_embed(self,
            x: tf.Tensor,
            ngram_size: int,
            weight_stddev: float = 0.01
            )-> tf.Tensor:
        pass


class ConvNgram(Decomposable, ABC):

    def ngram_embed(self,
            x: tf.Tensor,
            ngram_size: int,
            weight_stddev: float = 0.01,
            )-> tf.Tensor:
        with tf.device(self.device):
            t = tf.expand_dims(x, -1)
            t = tf.layers.conv2d(t, 
                    filters=int(t.shape[2]),
                    kernel_size=(ngram_size, t.shape[2]),
                    kernel_initializer=tf.initializers.truncated_normal(
                            stddev=weight_stddev),
                    name='%d-gram-conv' % ngram_size,
                    reuse=tf.AUTO_REUSE)
            t = tf.nn.relu(t)
            t = tf.squeeze(t, [2])
        return t


class TagGatedConvNgram(Decomposable, ABC):

    def __init__(self, tags_num: int)-> None:
        self.tags_num = tags_num

    def ngram_embed(self,
            x: tf.Tensor,
            ngram_size: int,
            weight_stddev: float = 0.01,
            )-> tf.Tensor:
        pass

