""" Prediction models.
"""

import typing as t

import numpy as np
import tensorflow as tf

import embed
import op
from nn.base import Model, SoftmaxCrossEntropyMixin
from util.log import exec_log as log
from util.debug import *


class AttentiveModel(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            project_dim: int = 500,
            ) -> None:
        super(AttentiveModel, self).__init__()
        self.project_dim = project_dim
        self._class_num = class_num

        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        def mask(x, x_len):
            # Explict mask the paddings.
            mask = tf.sequence_mask(x_len, tf.shape(x)[1], dtype=tf.float32)
            return tf.expand_dims(mask, -1)
        mask1, mask2 = mask(self.x1, self.len1), mask(self.x2, self.len2)

        with tf.variable_scope('embed') as s:
            embed = tf.constant(embeddings.get_embeddings(),
                                dtype=tf.float32,
                                name='embeddings')
            x1, x2 = map(lambda x: tf.gather(embed, x), [self.x1, self.x2])

        x1, x2 = map(lambda x: op.linear(x, scope='project', dim=500), [x1, x2])
        x1, x2 = map(lambda x: op.highway(x, scope='highway-1', dim=500), [x1, x2])
        x1, x2 = map(lambda x: op.highway(x, scope='highway-2', dim=500), [x1, x2])

        with tf.variable_scope('attent') as s:
            def soft_align(att_fn, scope):
                with tf.variable_scope(scope) as s:
                    sim = att_fn(x1, x2)
                    alpha = tf.matmul(tf.nn.softmax(tf.matrix_transpose(sim)), x1)
                    beta  = tf.matmul(tf.nn.softmax(sim), x2)
                    return alpha, beta
            a1, b1 = soft_align(self.attention_mul,  'mul')
            #a2, b2 = soft_align(self.attention_diff, 'diff')
            #a3, b3 = soft_align(self.attention_dist, 'dist')

        with tf.variable_scope('compare') as s:
            #v1 = self.forward(tf.concat([x1, b1], 2))
            #v2 = self.forward(tf.concat([x2, a1], 2))
            v1 = op.linear(tf.concat([x1, b1], 2), dim=500)
            v2 = op.linear(tf.concat([x2, a1], 2), dim=500)

        v1, v2 = map(lambda x: op.highway(x, scope='highway-3', dim=500), [v1, v2])
        v1, v2 = map(lambda x: op.highway(x, scope='highway-4', dim=500), [v1, v2])

        with tf.variable_scope('aggregate') as s:
            # CHANGE
            #def reduce_mean(x, x_len):
            #    return (tf.reduce_sum(x, axis=1) /
            #            tf.expand_dims(tf.cast(x_len, tf.float32), -1))
            def reduce_mean(x, x_len):
                return (tf.reduce_sum(x, axis=1) /
                        tf.cast(tf.shape(x)[1], tf.float32))
            v = tf.concat([
                    #reduce_mean(v1, self.len1),
                    #reduce_mean(v2, self.len2),
                    tf.reduce_max(v1, axis=1),
                    tf.reduce_max(v2, axis=1),
                    tf.reduce_sum(v1, axis=1),
                    tf.reduce_sum(v2, axis=1)
                    ], 1)
            y_hat = self.forward(v)
            y_hat = self.linear(y_hat, dim=self._class_num)

        self.evaluate_and_loss(y_hat)


    def linear(self, inputs: tf.Tensor, dim: int, bias=True):
        return op.linear(inputs, dim, activation_fn=None, bias=bias)


    def forward(self, 
            inputs: tf.Tensor,
            dim: int = None,
            scope: t.Union[str, tf.VariableScope] = None):
        scope = scope if scope else 'forward'
        kwargs = {'dim': dim if dim else self.project_dim,
                  'keep_prob': self.keep_prob,
                 }
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            t = op.linear(inputs, scope='linear-1', **kwargs)
            t = op.linear(t, scope='linear-2', **kwargs)
            return t


    def post_project(self, x1, x2):
        x1, x2 = map(lambda x: op.highway(x, scope='highway-1'), [x1, x2])
        x1, x2 = map(lambda x: op.highway(x, scope='highway-2'), [x1, x2])
        return x1, x2
        #return x1, x2


    def attention(self,
            x1: tf.Tensor,
            x2: tf.Tensor,
            scope: t.Union[str, tf.VariableScope] = None,
            ):
        """
        Inputs:  [batch, seq_len_1, embed_dim]
                 [batch, seq_len_2, embed_dim]
        Returns: [batch, seq_len_1, seq_len_2]
        """
        with tf.name_scope('attention') as s:
            att1 = self.attention_mul(x1, x2)
            #att1 = self.attention_dist(x1, x2)
            #return att1
            #att2 = self.attention_dist(x1, x2)
            return att1


    def attention_mul(self,
            x1: tf.Tensor,
            x2: tf.Tensor,
            scope: t.Union[str, tf.VariableScope] = None,
            ):
        """
        Inputs:  [batch, seq_len_1, embed_dim]
                 [batch, seq_len_2, embed_dim]
        Returns: [batch, seq_len_1, seq_len_2]
        """
        with tf.variable_scope('mul-att'):
            x1 = self.forward(x1)
            x2 = self.forward(x2)
            return tf.matmul(x1, tf.matrix_transpose(x2))


    def attention_dist(self,
            x1: tf.Tensor,
            x2: tf.Tensor,
            scope: t.Union[str, tf.VariableScope] = None,
            ):
        """
        Inputs:  [batch, seq_len_1, embed_dim]
                 [batch, seq_len_2, embed_dim]
        Returns: [batch, seq_len_1, seq_len_2]
        """
        with tf.variable_scope('dist-att') as s:
            v1 = tf.expand_dims(x1, 2)  # [batch, x1_len, 1, dim]
            v2 = tf.expand_dims(x2, 1)  # [batch, 1, x2_len, dim]
            edist = tf.reduce_sum(tf.square(v1 - v2), axis=3)
            # shape: [batch, x1_len, x2_len]
            att = 1 / (1 + edist)
            return att


    def attention_diff(self,
            x1: tf.Tensor,
            x2: tf.Tensor,
            scope: t.Union[str, tf.VariableScope] = None,
            ):
        """
        Inputs:  [batch, seq_len_1, embed_dim]
                 [batch, seq_len_2, embed_dim]
        Returns: [batch, seq_len_1, seq_len_2]
        """
        with tf.variable_scope('diff-att') as s:
            x1 = self.forward(x1)
            x2 = self.forward(x2)
            x1 = tf.expand_dims(x1, 2)
            x2 = tf.expand_dims(x2, 1)
            return tf.reduce_sum(1 / (1 + tf.abs(x1 - x2)), axis=-1)


    def intra(self, x1, x2):
        """ Intra-attention layer. """
        with tf.variable_scope('intra') as s:
            def attent(x):
                with tf.variable_scope('distance_bias', reuse=tf.AUTO_REUSE):
                    idx = tf.range(0, tf.shape(x)[1], 1)
                    dist = tf.abs(tf.expand_dims(idx, 0) - tf.expand_dims(idx, 1))
                    bias = tf.get_variable('bias', [1])
                    bias *= tf.cast(dist >= 10, tf.float32)
                    bias = tf.expand_dims(bias, 0)
                att = self.attention(x, x)
                pan = tf.nn.softmax(att + bias)
                xp = tf.einsum('bik,bkj->bij', pan, x)
                return tf.concat([x, xp], 2)
            return map(attent, [x1, x2])

