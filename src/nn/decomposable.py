from typing import Union, Callable

import numpy as np
import tensorflow as tf

import embed
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
            **kwargs,
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
            x1, x2 = map(self.post_project, [x1, x2])
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
            #def reduce_mean(x, x_len):
            #    return (tf.reduce_sum(x, axis=1) /
            #            tf.expand_dims(tf.cast(x_len, tf.float32), -1))
            #v1 = reduce_mean(v1 * mask1, self.len1)
            #v2 = reduce_mean(v2 * mask2, self.len2)
            y_hat = self.forward(tf.concat([v1, v2], 1))
            y_hat = self.linear(y_hat, dim=self._class_num)

        self.evaluate_and_loss(y_hat)


    def _linear(self,
            inputs: tf.Tensor,
            dim: int = -1,
            activation_fn: Callable = tf.nn.relu,
            weight_stddev: float = 0.01,
            bias: bool = True,
            bias_init: float = None,
            keep_prob: float = None,
            scope: Union[str, tf.VariableScope] = None,
            reuse: bool = tf.AUTO_REUSE,
            device: str = None
            ):
        """
        Inputs:  3D-Tensor [batch, seq_len, input_dim], or
                 2D-Tensor [batch, input_dim]
        Returns: 3D-Tensor [batch, seq_len, dim], or
                 2D-Tensor [batch, dim]
        """
        keep_prob = keep_prob if keep_prob else self.keep_prob
        device = device if device else self.device
        dim = dim if dim > 0 else int(inputs.shape[-1])
        t = tf.nn.dropout(inputs, keep_prob)
        with tf.device(device):
            with tf.variable_scope(scope if scope else 'linear', reuse=reuse):
                t_shape = tf.shape(t)
                w = tf.get_variable('weight',
                        shape=[t.get_shape().as_list()[-1], dim],
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal(
                                stddev=weight_stddev))
                output_rank = len(t.get_shape())
                if output_rank == 3:
                    t = tf.reshape(t, [-1, t.shape[2]])
                t = tf.matmul(t, w)
                if bias:
                    if bias_init is None:
                        bias_init = self.bias_init
                    if bias_init == 0:
                        init = tf.initializers.zeros()
                    else:
                        init = tf.initializers.constant(bias_init)
                    b = tf.get_variable('bias',
                            shape=[dim],
                            dtype=tf.float32,
                            initializer=init)
                    t += b
                if output_rank == 3:
                    t = tf.reshape(t, [-1, t_shape[1], dim])
                t = activation_fn(t) if activation_fn else t
        return t


    def linear(self, inputs: tf.Tensor, dim: int, bias=True):
        return self._linear(inputs, dim,
                keep_prob=1.0,
                activation_fn=None,
                bias=bias)


    def forward(self, 
            inputs: tf.Tensor,
            scope: Union[str, tf.VariableScope] = None,
            ):
        scope = scope if scope else 'forward'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            t = self._linear(inputs, self.project_dim, scope='linear-1')
            t = self._linear(t, scope='linear-2')
            return t


    def post_project(self, x):
        return x


    def attention(self,
            x1: tf.Tensor,
            x2: tf.Tensor,
            scope: Union[str, tf.VariableScope] = None,
            ):
        """
        Inputs:  [batch, seq_len_1, embed_dim]
                 [batch, seq_len_2, embed_dim]
        Returns: [batch, seq_len_1, seq_len_2]
        """
        with tf.name_scope('attention') as s:
            #sim = (tf.expand_dims(self.forward(x1), 2) *
            #       tf.expand_dims(self.forward(x2), 1))
            #sim = tf.reduce_sum(sim, axis=3)
            x1 = self.forward(x1)
            x2 = self.forward(x2)
            return tf.matmul(x1, tf.matrix_transpose(x2))


    def intra(self, x1, x2):
        """ Intra-attention layer. """
        with tf.variable_scope('intra') as s:
            with tf.name_scope('distance_bias'):
                idx = tf.range(0, self.seq_len, 1)
                dist = tf.abs(tf.expand_dims(idx, 0) - tf.expand_dims(idx, 1))
                bias = tf.get_variable('bias', [1])
                bias *= tf.cast(dist >= 10, tf.float32)
                bias = tf.expand_dims(bias, 0)
            def attent(x):
                att = self.attention(x, x)
                xp = tf.einsum('bik,bkj->bij', tf.nn.softmax(att + bias), x)
                return tf.concat([x, xp], 2)
            return map(attent, [x1, x2])

