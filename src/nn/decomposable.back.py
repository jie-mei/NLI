import numpy as np
import tensorflow as tf

from nn.base import Model, SoftmaxCrossEntropyMixin
import op


class Decomposeable(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            seq_len: int,
            class_num: int,
            word_embeddings: np.ndarray,
            project_dim: int = 200,
            attent_dim: int = -1,
            keep_prob: float = 1.0,
            intra_attention: bool = False,
            **kwargs,
            ) -> None:
        self.seq_len = seq_len
        self.keep_prob = keep_prob
        self.attent_dim = attent_dim
        self.project_dim = project_dim

        self.x1 = tf.placeholder(tf.int32, name='x1', shape=[None, seq_len])
        self.x2 = tf.placeholder(tf.int32, name='x2', shape=[None, seq_len])
        self.y  = tf.placeholder(tf.int32, name='y',  shape=[None])

        # Feed-forward setups for ReLU layers
        self._fc_opts = dict(activation_fn=tf.nn.relu,
                             keep_prob=keep_prob,
                             reuse=tf.AUTO_REUSE)

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
            #with tf.variable_scope('project') as s:
            project = lambda x: op.fc(x, dim=self.project_dim, scope=s,
                    reuse=tf.AUTO_REUSE)
            x1, x2 = map(lambda x: project(x), [x1, x2])

        with tf.variable_scope('attent') as s:
            sim = op.sim.forward_product(x1, x2, scope=s, dim=self.attent_dim, **self._fc_opts)
            sim *= mask1 * tf.matrix_transpose(mask2)
            alpha = tf.einsum('bki,bkj->bij', op.softmax(sim, axis=1), x1)
            beta  = tf.einsum('bik,bkj->bij', op.softmax(sim, axis=2), x2)
        
        with tf.variable_scope('compare') as s:
            v1 = op.fc(tf.concat([x1, beta ], 2), scope=s, **self._fc_opts)
            v2 = op.fc(tf.concat([x2, alpha], 2), scope=s, **self._fc_opts)

        with tf.variable_scope('aggregate') as s:
            v1 = tf.reduce_sum(v1 * mask1, axis=1)
            v2 = tf.reduce_sum(v2 * mask2, axis=1)
            y_hat = op.fc(tf.concat([v1, v2], 1), dim=class_num, scope=s)

        self.evaluate_and_loss(y_hat)

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
                att = op.sim.forward_product(x, x, scope=s, dim=self.attent_dim,
                        **self._fc_opts)
                xp = tf.einsum('bik,bkj->bij', op.softmax(att + bias, 2), x)
                return tf.concat([x, xp], 2)
            return map(attent, [x1, x2])

