import numpy as np
import tensorflow as tf

from nn.base import Model, SoftmaxCrossEntropyMixin
import op

import data
from typing import Union, Callable


class Decomposeable(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            dataset: data.Dataset,
            class_num: int,
            project_dim: int = 200,
            keep_prob: float = 1.0,
            intra_attention: bool = False,
            **kwargs,
            ) -> None:
        self.keep_prob = keep_prob
        self.project_dim = project_dim
        self.intra_attention = intra_attention
        self._class_num = class_num

        # Fit the given dataset with optional parameters.
        self.feed(dataset, **kwargs)

        def mask(x, x_len):
            # Explict masking the paddings.
            size = tf.reshape(x_len, [-1])
            mask = tf.sequence_mask(size, tf.shape(x)[1], dtype=tf.float32)
            return tf.expand_dims(mask, -1)
        mask1, mask2 = mask(self.x1, self.len1), mask(self.x2, self.len2)

        with tf.variable_scope('embed') as s:
            embed = lambda x: op.embedding(x, dataset.embeddings(), normalize=True)
            x1, x2 = map(embed, [self.x1, self.x2])
            project = lambda x: self.linear(x, self.project_dim)
            x1, x2 = map(project, [x1, x2])
            x1, x2 = self.intra(x1, x2) if intra_attention else (x1, x2)

        with tf.variable_scope('attent') as s:
            sim = self.attention(x1, x2)
            sim *= mask1 * tf.matrix_transpose(mask2)
            #alpha = tf.einsum('bki,bkj->bij', op.softmax(sim, axis=2), x1)
            #beta  = tf.einsum('bik,bkj->bij', op.softmax(sim, axis=1), x2)
            alpha = tf.matmul(tf.nn.softmax(tf.matrix_transpose(sim)), x1)
            beta  = tf.matmul(tf.nn.softmax(sim), x2)
        
        with tf.variable_scope('compare') as s:
            v1 = self.forward(tf.concat([x1, beta ], 2))
            v2 = self.forward(tf.concat([x2, alpha], 2))

        with tf.variable_scope('aggregate') as s:
            v1 = tf.reduce_sum(v1 * mask1, axis=1)
            v2 = tf.reduce_sum(v2 * mask2, axis=1)
            y_hat = self.forward(tf.concat([v1, v2], 1))
            y_hat = self.linear(y_hat, dim=class_num, bias=True)

        self.evaluate_and_loss(y_hat)


    def _linear(self,
            inputs: tf.Tensor,
            dim: int = -1,
            keep_prob: float = None,
            activation_fn: Callable = tf.nn.relu,
            weight_stddev: float = 0.01,
            bias: bool = True,
            scope: Union[str, tf.VariableScope] = None,
            reuse: bool = tf.AUTO_REUSE,
            ):
        """
        Inputs:  [batch, seq_len, input_dim]
        Returns: [batch, seq_len, dim]
        """
        scope = scope if scope else 'linear'
        keep_prob = keep_prob if keep_prob else self.keep_prob
        dim = dim if dim > 0 else int(inputs.shape[-1])
        bias_init = tf.constant_initializer(0) if bias else None
        t = tf.nn.dropout(inputs, keep_prob)
        t = tf.contrib.layers.fully_connected(t, dim,
                activation_fn=None,
                weights_initializer=tf.initializers.truncated_normal(
                        stddev=weight_stddev),
                biases_initializer=bias_init,
                scope=scope,
                reuse=reuse)
        t = activation_fn(t) if activation_fn else t
        return t


    def linear(self, inputs: tf.Tensor, dim: int, bias=False):
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
            sim = (tf.expand_dims(self.forward(x1), 2) *
                   tf.expand_dims(self.forward(x2), 1))
            sim = tf.reduce_sum(sim, axis=3)
            return sim


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
                xp = tf.einsum('bik,bkj->bij', op.softmax(att + bias, 2), x)
                return tf.concat([x, xp], 2)
            return map(attent, [x1, x2])

