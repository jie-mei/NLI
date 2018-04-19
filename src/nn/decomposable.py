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
            dataset=None,
            **kwargs,
            ) -> None:
        self.seq_len = seq_len
        self.keep_prob = keep_prob
        self.project_dim = project_dim
        self.intra_attention = intra_attention

        self.x1, self.x2, self.y = dataset.x1, dataset.x2, dataset.y

        def mask(x):
            # Explict masking the paddings.
            size = tf.count_nonzero(x, axis=1, dtype=tf.float32)
            mask = tf.sequence_mask(size, self.seq_len, dtype=tf.float32)
            return tf.expand_dims(mask, -1), size
        s1, s2 = map(mask, [self.x1, self.x2])
        mask1, size1 = s1
        mask2, size2 = s2

        with tf.variable_scope('embed') as s:
            embed = lambda x: op.embedding(x, word_embeddings, normalize=True)
            x1, x2 = map(lambda x: embed(x), [self.x1, self.x2])
#            project = lambda x: self.linear(x, self.project_dim)
#            x1, x2 = map(lambda x: project(x), [x1, x2])
            x1, x2 = self.intra(x1, x2) if intra_attention else (x1, x2)

        with tf.variable_scope('attent') as s:
            sim = self.attention(x1, x2)
            sim *= mask1 * tf.matrix_transpose(mask2)
            alpha = tf.matmul(tf.nn.softmax(tf.transpose(sim, [0, 2, 1])), x1)
            beta = tf.matmul(tf.nn.softmax(sim), x2)
        
        with tf.variable_scope('compare') as s:
#            v1 = self.conditionalBN(x1, beta)
#            v2 = self.conditionalBN(x2, alpha)
            v1 = self.forward(tf.concat([x1, beta ], 2))
            v2 = self.forward(tf.concat([x2, alpha], 2))

        with tf.variable_scope('aggregate') as s:
            v1 = tf.reduce_sum(v1, axis=1)
            v2 = tf.reduce_sum(v2, axis=1)
            y_hat = self.forward(tf.concat([v1, v2], 1))
            y_hat = self._linear(y_hat, dim=class_num, activation_fn=None)

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
        t = inputs
        t = tf.contrib.layers.fully_connected(t, dim,
                activation_fn=None,
                weights_initializer=tf.initializers.truncated_normal(
                        stddev=weight_stddev),
                biases_initializer=bias_init,
                scope=scope,
                reuse=reuse)
        t = activation_fn(t) if activation_fn else t
        return t


    def linear(self, inputs: tf.Tensor, dim: int):
        return self._linear(inputs, dim,
                keep_prob=1.0,
                activation_fn=None,
                bias=False)


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

    def conditionalBN(self, x1, x2):
        mean, variance = tf.nn.moments(x1, [0, 1])
        num_outputs = x1.get_shape().as_list()[-1]
        betas = tf.contrib.layers.fully_connected(
            inputs=x2,
            num_outputs=num_outputs,
            activation_fn=None,
            reuse=tf.AUTO_REUSE,
            scope='conditional_betas'
        )
        gammas = tf.contrib.layers.fully_connected(
            inputs=x2,
            num_outputs=num_outputs,
            activation_fn=None,
            reuse=tf.AUTO_REUSE,
            scope='_conditional_gammas'
        )
        inv = gammas * tf.expand_dims(tf.rsqrt(variance + tf.keras.backend.epsilon()), 0)
        out = inv * (x1 - mean) + betas
        out = tf.nn.relu(out)
        return out
