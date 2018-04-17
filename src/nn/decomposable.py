import numpy as np
import tensorflow as tf

from nn.base import Model, SoftmaxCrossEntropyMixin
import op

from typing import Union, Callable


class Decomposeable(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            seq_len: int,
            class_num: int,
            word_embeddings: np.ndarray,
            project_dim: int = 200,
            keep_prob: float = 1.0,
            intra_attention: bool = False,
            **kwargs,
            ) -> None:
        self.seq_len = seq_len
        self.keep_prob = keep_prob
        self.project_dim = project_dim
        self.intra_attention = intra_attention

        self.x1 = tf.placeholder(tf.int32, name='x1', shape=[None, seq_len])
        self.x2 = tf.placeholder(tf.int32, name='x2', shape=[None, seq_len])
        self.y  = tf.placeholder(tf.int32, name='y',  shape=[None])

        def mask(x):
            # Explict masking the paddings.
            size = tf.count_nonzero(x, axis=1, dtype=tf.float32)
            mask = tf.sequence_mask(size, self.seq_len, dtype=tf.float32)
            return tf.expand_dims(mask, -1)
        mask1, mask2 = map(mask, [self.x1, self.x2])

        with tf.variable_scope('embed') as s:
            embed = lambda x: op.embedding(x, word_embeddings, normalize=True)
            x1, x2 = map(lambda x: embed(x), [self.x1, self.x2])
            x1, x2 = self.intra(x1, x2) if intra_attention else (x1, x2)
            x1 = self.forward(x1, dim=self.project_dim, scope='project_x1')
            x2 = self.forward(x2, dim=self.project_dim, scope='project_x2')

        with tf.variable_scope('attent') as s:
            sim = self.attention(x1, x2)
            sim *= mask1 * tf.matrix_transpose(mask2)
            alpha = tf.einsum('bki,bkj->bij', op.softmax(sim, axis=1), x1)
            beta  = tf.einsum('bik,bkj->bij', op.softmax(sim, axis=2), x2)
        
        with tf.variable_scope('compare') as s:
            v1 = self.forward(tf.concat([x1, beta ], 2), scope='compare_x1')
            v2 = self.forward(tf.concat([x2, alpha], 2), scope='compare_x2')

        with tf.variable_scope('aggregate') as s:
            v1 = tf.reduce_sum(v1 * mask1, axis=1)
            v2 = tf.reduce_sum(v2 * mask2, axis=1)
            y_hat = self.forward(tf.concat([v1, v2], 1), dim=class_num)

        self.evaluate_and_loss(y_hat)


    def forward(self,
            inputs: tf.Tensor,
            dim: int = -1,
            activation_fn: Callable = tf.nn.relu,
            weight_stddev: float = 0.01,
            scope: Union[str, tf.VariableScope] = None,
            reuse: bool = False,
            ):
        """
        Inputs:  [batch, seq_len, input_dim]
        Returns: [batch, seq_len, dim]
        """
        scope = scope if scope else 'forward'
        dim = dim if dim > 0 else int(inputs.shape[-1])
        t = tf.nn.dropout(inputs, self.keep_prob)
        t = tf.contrib.layers.fully_connected(inputs, dim,
                activation_fn=None,
                weights_initializer=tf.initializers.truncated_normal(
                        stddev=weight_stddev),
                biases_initializer=None,
                scope=scope,
                reuse=reuse)
        t = activation_fn(t) if activation_fn else t
        return t


    def attention(self,
            x1: tf.Tensor,
            x2: tf.Tensor,
            scope: Union[str, tf.VariableScope] = None,
            **kwargs,
            ):
        """
        Inputs:  [batch, seq_len_1, embed_dim]
                 [batch, seq_len_2, embed_dim]
        Returns: [batch, seq_len_1, seq_len_2]
        """
        with tf.name_scope('attention') as s:
            sim = (tf.expand_dims(self.forward(x1, **kwargs), 2) *
                   tf.expand_dims(self.forward(x2, **kwargs), 1))
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

