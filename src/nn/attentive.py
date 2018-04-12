""" Prediction models.
"""

from abc import ABC, abstractmethod
import sys
from typing import List, Union, Tuple

import tensorflow as tf
import numpy as np

import nn
import attention
import encode
import evaluate
from util.annotation import name_scope
from util.display import ReprMixin


class AttentiveModel(nn.Model):
    """ A basic NN model to conduct pairwised text analysis.

    Attributes:
        text_max_len: The maximum number of words in a text.
        filter_width: The width of the filter.
    """
    def __init__(self,
                 text_max_len: int,
                 word_embeddings: np.ndarray,
                 visualize_attention_softmax: bool = False,
                 visualize_saliency_softmax: bool = False,
                 attention: str = 'NoAttention',
                 encoder: str = 'NoEncode',
                 evaluator: str = 'CosineMSE',
                 **kwargs
                 ) -> None:
        self.text_max_len = text_max_len

        input_shape = [None, self.text_max_len]
        self.x1 = tf.placeholder(tf.int32, name="x1", shape=input_shape)
        self.x2 = tf.placeholder(tf.int32, name="x2", shape=input_shape)
        self.y = tf.placeholder(tf.int32, name="y", shape=[None])

        def count_seq_len(x):
            return tf.count_nonzero(x, 1, keep_dims=True, dtype=tf.float32)
        self.len1, self.len2 = map(count_seq_len, [self.x1, self.x2])

        embed1 = nn.layer.embedding(self.x1, word_embeddings)
        embed2 = nn.layer.embedding(self.x2, word_embeddings)

        with tf.name_scope('encode'):
            self.encoder = self._init_component(encode, encoder, 'encode_', kwargs)
            #x1, x2 = self.encoder.encode(embed1, embed2)
            embed1, embed2 = self.encoder.encode(embed1, embed2)

        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
            self.attention = self._init_component(sys.modules['attention'],
                    attention, 'attention_', kwargs)
            x1, x2, alpha1, alpha2 = self.attention.attent(embed1, embed2)

        with tf.variable_scope('evaluate', reuse=tf.AUTO_REUSE):
            def text_embeddings(x, x_len):
                return tf.reduce_sum(x, axis=2) / x_len
            x1 = text_embeddings(x1, self.len1)
            x2 = text_embeddings(x2, self.len2)

            self.evaluator = self._init_component(evaluate, evaluator, 'evaluate_',
                    kwargs)
            # self.prediction is discrete and thus non-differentiable, we need to keep the probability for saliency map
            self.prediction, self.probability = self.evaluator.predict(x1, x2)
            self.performance = self.evaluator.evaluate(self.prediction, self.y)
        self.loss = self.evaluator.loss(x1, x2)

        with tf.name_scope('visualize'):
            with tf.name_scope('activation'):
                if len(alpha1.shape) > 2 and int(alpha1.shape[1]) > 1:
                    alpha1 = tf.reduce_sum(alpha1, axis=1, keep_dims=True)
                    alpha2 = tf.reduce_sum(alpha2, axis=1, keep_dims=True)
                self.x1_att, self.x2_att = alpha1, alpha2
                if visualize_attention_softmax:
                    self.x1_att, self.x2_att = map(tf.contrib.layers.softmax,
                                                   [self.x1_att, self.x2_att])
            with tf.name_scope('saliency'):
                def saliency_map(x_embed):
                    dy_dx = tf.gradients(ys=self.probability, xs=x_embed)
                    sal = tf.reduce_mean(dy_dx[0], axis=1)
                    return sal / tf.reduce_sum(sal, axis=1, keep_dims=True)
                self.x1_sal, self.x2_sal = map(saliency_map, [embed1, embed2])
                if visualize_saliency_softmax:
                    self.x1_sal, self.x2_sal = map(tf.contrib.layers.softmax,
                                                   [self.x1_sal, self.x2_sal])
