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

    def post_project(self, x1: tf.Tensor, x2: tf.Tensor):
        for size in self.ngram_size:
            x1_ngram, x2_ngram = self.ngram_embed(x1, x2, size)
            x1 = tf.concat([x1, x1_ngram], 1)
            x2 = tf.concat([x2, x2_ngram], 1)
        return x1, x2

    @abstractmethod
    def ngram_embed(self,
            x1: tf.Tensor,
            x2: tf.Tensor,
            ngram_size: int,
            weight_stddev: float = 0.01
            )-> tf.Tensor:
        pass


class ConvNgram(Ngram):

    def ngram_embed(self,
            x1: tf.Tensor,
            x2: tf.Tensor,
            ngram_size: int,
            weight_stddev: float = 0.01,
            )-> tf.Tensor:
        def ngram_embed_impl(x):
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
        return map(ngram_embed_impl, [x1, x2])


class TagConvNgram(ConvNgram):

    def __init__(self, tags_num: int = 45, **kwargs)-> None:
        self.tags_num = tags_num
        super(TagConvNgram, self).__init__(**kwargs)

    def post_project(self, x1: tf.Tensor, x2: tf.Tensor):
        tag_weight = tf.get_variable('tag_weight',
                shape=(self.tags_num, 100),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal(stddev=0.01))
                #initializer=tf.initializers.constant(1))
        x1 = tf.concat([x1, tf.gather(tag_weight, self.tag1)], 2)
        x2 = tf.concat([x2, tf.gather(tag_weight, self.tag2)], 2)
        x1_ngrams, x2_ngrams = [x1], [x2]
        for size in self.ngram_size:
            with tf.name_scope('%d-gram-embed' % size):
                x1_ngram, x2_ngram = self.ngram_embed(x1, x2, size)
                x1_ngrams += x1_ngram,
                x2_ngrams += x2_ngram,
        x1 = tf.concat(x1_ngrams, 1)
        x2 = tf.concat(x2_ngrams, 1)
        return x1, x2


class TagGatedConvNgram(ConvNgram):

    def __init__(self, tags_num: int = 45, **kwargs)-> None:
        self.tags_num = tags_num
        super(TagGatedConvNgram, self).__init__(**kwargs)

    def post_project(self, x1: tf.Tensor, x2: tf.Tensor):
        tag_weight = tf.get_variable('tag_weight',
                shape=(self.tags_num, 1),
                dtype=tf.float32,
                #initializer=tf.initializers.truncated_normal(stddev=0.01))
                initializer=tf.initializers.constant(1))
        x1 *= tf.gather(tag_weight, self.tag1)
        x2 *= tf.gather(tag_weight, self.tag2)
        for size in [n for n in self.ngram_size if n > 1]:
            x1_ngram, x2_ngram = self.ngram_embed(x1, x2, size)
            x1 = tf.concat([x1, x1_ngram], 1)
            x2 = tf.concat([x2, x2_ngram], 1)
        return x1, x2
