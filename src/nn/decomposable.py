import typing as t

import numpy as np
import tensorflow as tf

import embed
import op
from nn.base import Model, SoftmaxCrossEntropyMixin
from util.log import exec_log as log


class Decomposable(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            project_dim: int = 200,
            intra_attention: bool = False,
            device: str = 'gpu:1',
            bias_init: float = 0,
            ) -> None:
        super(Decomposable, self).__init__()
        self.project_dim = project_dim
        self.intra_attention = intra_attention
        self.device = device
        self.bias_init = bias_init
        self._class_num = class_num

        log.debug('Model train on device %s' % self.device)

        with tf.device(self.device):
            self.keep_prob = tf.placeholder(tf.float32, shape=[])

        def mask(x, x_len):
            # Explict mask the paddings.
            mask = tf.sequence_mask(x_len, tf.shape(x)[1], dtype=tf.float32)
            return tf.expand_dims(mask, -1)
        # mask1, mask2 = mask(self.x1, self.len1), mask(self.x2, self.len2)

        with tf.variable_scope('embed') as s:
            with tf.device(self.device):
                embed = tf.constant(embeddings.get_embeddings(),
                                    dtype=tf.float32,
                                    name='embeddings')
            x1, x2 = map(lambda x: tf.gather(embed, x), [self.x1, self.x2])
            # Linear projection
            project = lambda x: self.linear(x, self.project_dim, bias=False)
            x1, x2 = map(project, [x1, x2])
            # Post-projection processing
            x1, x2 = self.post_project(x1, x2)
            x1, x2 = self.intra(x1, x2) if intra_attention else (x1, x2)

        with tf.variable_scope('attent') as s:
            sim = self.attention(x1, x2)
            # sim *= mask1 * tf.matrix_transpose(mask2)
            alpha = tf.matmul(tf.nn.softmax(tf.matrix_transpose(sim)), x1)
            beta  = tf.matmul(tf.nn.softmax(sim), x2)
        
        with tf.variable_scope('compare') as s:
            v1 = self.forward(tf.concat([x1, beta ], 2))
            v2 = self.forward(tf.concat([x2, alpha], 2))

        with tf.variable_scope('aggregate') as s:
            # CHANGE
            v1 = tf.reduce_sum(v1, axis=1)
            v2 = tf.reduce_sum(v2, axis=1)
            y_hat = self.forward(tf.concat([v1, v2], 1))
            #def reduce_mean(x, x_len):
            #    return (tf.reduce_sum(x, axis=1) /
            #            tf.expand_dims(tf.cast(x_len, tf.float32), -1))
            #v1 = reduce_mean(v1 * mask1, self.len1)
            #v2 = reduce_mean(v2 * mask2, self.len2)
            y_hat = self.linear(y_hat, dim=self._class_num)

        self.evaluate_and_loss(y_hat)


    def linear(self, inputs: tf.Tensor, dim: int, bias=True):
        return op.linear(inputs, dim,
                keep_prob=1.0,
                activation_fn=None,
                bias=bias)


    def forward(self, 
            inputs: tf.Tensor,
            scope: t.Union[str, tf.VariableScope] = None,
            ):
        scope = scope if scope else 'forward'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            t = op.linear(inputs, self.project_dim, scope='linear-1')
            t = op.linear(t, scope='linear-2')
            return t


    def post_project(self, x1, x2):
        return x1, x2


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
            x1 = self.forward(x1)
            x2 = self.forward(x2)
            return tf.matmul(x1, tf.matrix_transpose(x2))


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

