""" Prediction models.
"""

import typing as t

import numpy as np
import tensorflow as tf
import tensorpack as tp

import embed
import op
from nn.base import Model, SoftmaxCrossEntropyMixin
from util.log import exec_log as log
from util.debug import *


class ResModel(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            project_dim: int,
            ) -> None:
        super(ResModel, self).__init__()
        self._class_num = class_num
        self.project_dim = project_dim
        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        def mask(x, x_len):
            # Explict mask the paddings.
            mask = tf.sequence_mask(x_len, tf.shape(x)[1], dtype=tf.float32)
            return tf.expand_dims(mask, -1)
        # mask1, mask2 = mask(self.x1, self.len1), mask(self.x2, self.len2)

        with tf.variable_scope('embed') as s:
            embed = tf.constant(embeddings.get_embeddings(),
                                dtype=tf.float32,
                                name='embeddings')
            x1, x2 = map(lambda x: tf.gather(embed, x), [self.x1, self.x2])

        for i in range(1, 11):
            x1, x2 = self.layer(x1, x2, 'res-layer-%d' % i)

        with tf.variable_scope('aggregate') as s:
            v = tf.concat([
                    tf.reduce_max(x1, axis=1),
                    tf.reduce_max(x2, axis=1),
                    tf.reduce_sum(x1, axis=1),
                    tf.reduce_sum(x2, axis=1)
                    ], 1)
            y_hat = self.forward(v)
            y_hat = op.linear(y_hat, dim=self._class_num, activation_fn=None)

        self.evaluate_and_loss(y_hat)

    def block(self, x1, x2):
        with tf.name_scope('attent'):
            v1 = tf.expand_dims(x1, 2)  # [batch, x1_len, 1, dim]
            v2 = tf.expand_dims(x2, 1)  # [batch, 1, x2_len, dim]
            edist = tf.reduce_sum(tf.square(v1 - v2), axis=3)
            # shape: [batch, x1_len, x2_len]
            att = 1 / (1 + edist)
        # sim *= mask1 * tf.matrix_transpose(mask2)
        with tf.name_scope('align'):
            alpha = tf.matmul(tf.nn.softmax(tf.matrix_transpose(att)), x1)
            beta  = tf.matmul(tf.nn.softmax(att), x2)
        x1 = tf.concat([x1, beta],  2)
        x2 = tf.concat([x2, alpha], 2)
        return x1, x2

    def layer(self, x1, x2, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            def linear_BNReLU(x, scope):
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    x = op.linear(x, dim=self.project_dim, activation_fn=None)
                    x = tf.layers.batch_normalization(x)
                    x = tf.nn.relu(x)
                    return x
            x1, x2 = map(lambda x: linear_BNReLU(x, 'linear-BN-ReLU-1'), [x1, x2])
            v1, v2 = self.block(x1, x2)
            v1, v2 = map(lambda x: linear_BNReLU(x, 'linear-BN-ReLU-2'), [v1, v2])
            def transit(x, v):
                gate = op.linear(x, activation_fn=tf.sigmoid, scope='gate')
                return v * gate + x * (1 - gate)
            return transit(x1, v1), transit(x2, v2)

    def forward(self, 
            inputs: tf.Tensor,
            dim: int = None,
            scope: t.Union[str, tf.VariableScope] = None):
        scope = scope if scope else 'forward'
        kwargs = {'dim': dim if dim else -1,
                  'keep_prob': self.keep_prob,
                  'weight_init': tf.truncated_normal_initializer(stddev=0.01)}
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            t = op.linear(inputs, scope='linear-1', **kwargs)
            t = op.linear(t, scope='linear-2', **kwargs)
            return t
