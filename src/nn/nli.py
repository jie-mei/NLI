import typing as t

import numpy as np
import tensorflow as tf

import embed
from nn.decomposable import Decomposable


class Ngram(Decomposable):

    def post_project(self,
            x: tf.Tensor,
            ngram_size: t.List[int] = [2, 3]):
        for size in ngram_size:
            x = tf.concat([x, self.ngram_embed(x, size)], 1)
        return x

    def ngram_embed(self,
            x: tf.Tensor,
            ngram_size: int,
            weight_stddev: float = 0.01,
            ):
        with tf.device(self.device):
            t = tf.expand_dims(x, -1)
            t = tf.layers.conv2d(t, 
                    filters=int(t.shape[2]),
                    kernel_size=(ngram_size, t.shape[2]),
                    kernel_initializer=tf.initializers.truncated_normal(
                            stddev=weight_stddev),
                    name='%d-gram-conv' % ngram_size,
                    reuse=tf.AUTO_REUSE)
            t = tf.squeeze(t, [2])
        return t

