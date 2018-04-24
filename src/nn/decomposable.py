import numpy as np
import tensorflow as tf

from nn.base import Model, SoftmaxCrossEntropyMixin
import op

from typing import Union, Callable


class Decomposeable(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            class_num: int,
            word_embeddings: np.ndarray,
            project_dim: int = 200,
            keep_prob: float = 1.0,
            intra_attention: bool = False,
            dataset=None,
            train_mode=True,
            **kwargs,
            ) -> None:
        self.seq_len = dataset.max_len
        self.keep_prob = keep_prob
        self.project_dim = project_dim
        self.intra_attention = intra_attention
        self.train_mode = train_mode

        self.x1, self.x2, self.y = dataset.x1, dataset.x2, dataset.y


        def mask(x):
            # Explict masking the paddings.
            size = tf.count_nonzero(x, axis=1, dtype=tf.float32)
            mask = tf.sequence_mask(size, self.seq_len, dtype=tf.float32)
            return tf.expand_dims(mask, -1)
        mask1, mask2 = map(mask, [self.x1, self.x2])

        with tf.variable_scope('embed') as s:
            embed = lambda x: op.embedding(x, word_embeddings, normalize=False)
            x1, x2 = map(lambda x: embed(x), [self.x1, self.x2])
            project = lambda x: self.linear(x, self.project_dim)
            x1, x2 = map(lambda x: project(x), [x1, x2])

        with tf.variable_scope('attent') as s:
            sim = self.attention(x1, x2)
            sim *= mask1 * tf.matrix_transpose(mask2)
            alpha = tf.matmul(tf.nn.softmax(tf.transpose(sim, [0, 2, 1])), x1)
            beta = tf.matmul(tf.nn.softmax(sim), x2)
        
        with tf.variable_scope('compare') as s:
            num_layers = 2
            num_nodes = [200] * num_layers
            # num_nodes = [150] * num_layers
            activations = [tf.nn.relu] * num_layers
            biases = [True] * num_layers
            v1 = self.forward(tf.concat([x1, beta ], 2), num_nodes, activations, biases)
            v2 = self.forward(tf.concat([x2, alpha], 2), num_nodes, activations, biases)

        with tf.variable_scope('aggregate') as s:
            v1 = tf.reduce_sum(v1 * mask1, axis=1)
            v2 = tf.reduce_sum(v2 * mask2, axis=1)

            # v1 = tf.reduce_sum(v1, axis=1)
            # v2 = tf.reduce_sum(v2, axis=1)
            # y_hat = self.forward(tf.concat([v1, v2], 1), num_nodes=[200, 200, class_num], activations=[tf.nn.relu, tf.nn.relu, None], biases=[True, True, True])
            # y_hat = self.forward(tf.concat([v1, v2], 1), num_nodes=[100, class_num], activations=[tf.nn.relu,  None], biases=[ True, True])
            # y_hat = self.forward(tf.concat([v1, v2], 1), num_nodes=[100, class_num], activations=[tf.nn.relu,  None], biases=[True, True])
            # y_hat = self.forward(tf.concat([v1, v2], 1), num_nodes=[class_num], activations=[None], biases=[True])

            x_concat = tf.concat([v1, v2], 1)

            # with tf.variable_scope('maxout') as s:
                # x_concat = self.maxout(x_concat, 3, 100) 

#            with tf.variable_scope('label1'):
#                y1 = self.forward(self.feedforward_attention(x_concat), num_nodes=[100, 1], activations=[tf.nn.relu,  None], biases=[True, True])
#            with tf.variable_scope('label2'):
#                y2 = self.forward(self.feedforward_attention(x_concat), num_nodes=[100, 1], activations=[tf.nn.relu,  None], biases=[True, True])
#            with tf.variable_scope('label3'):
#                y3 = self.forward(self.feedforward_attention(x_concat), num_nodes=[100, 1], activations=[tf.nn.relu,  None], biases=[True, True])
#            y_hat = tf.concat([y1, y2, y3], axis=1)
            y_hat = self.forward(x_concat, num_nodes=[100, class_num], activations=[tf.nn.relu,  None], biases=[True, True])
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
            regularizer = tf.contrib.layers.l2_regularizer(0.0001)
            ):
        """
        Inputs:  [batch, seq_len, input_dim]
        Returns: [batch, seq_len, dim]
        """
        scope = scope if scope else 'linear'
        keep_prob = keep_prob if keep_prob else self.keep_prob
        dim = dim if dim > 0 else int(inputs.shape[-1])
        bias_init = tf.constant_initializer(0) if bias else None
        t = tf.layers.dropout(inputs, 1 - keep_prob, training=self.train_mode)
        t = inputs
        t = tf.contrib.layers.fully_connected(t, dim,
                activation_fn=activation_fn,
                weights_initializer=tf.initializers.truncated_normal(
                        stddev=weight_stddev),
                biases_initializer=bias_init,
                weights_regularizer=regularizer,
                biases_regularizer=regularizer,
                scope=scope,
                reuse=reuse)
        return t


    def linear(self, inputs: tf.Tensor, dim: int):
        return self._linear(inputs, dim,
                keep_prob=1.0,
                # activation_fn=tf.nn.relu,
                activation_fn=None,
                bias=True)


    def forward(self, 
            inputs: tf.Tensor,
            num_nodes=[-1],
            activations=[tf.nn.relu],
            biases=[True],
            scope: Union[str, tf.VariableScope] = None,
            noise=True
            ):
        scope = scope if scope else 'forward'
        t = inputs
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for i, (dim, activation, bias) in enumerate(zip(num_nodes, activations, biases)):
                t = self._linear(t, dim=dim, scope='linear-{}'.format(i), activation_fn=activation, bias=bias)
                if noise and self.train_mode:
                    t = self.gaussian_noise(t)
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
            new_x1 = self.forward(x1, biases=[True], activations=[tf.nn.relu])
            new_x2 = self.forward(x2, biases=[True], activations=[tf.nn.relu])
            sim = tf.matmul(new_x1, tf.transpose(new_x2, [0, 2, 1]))
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

    def maxout(self, inputs, num_channels=3, output_dim=100):
        components = []
        for i in range(num_channels):
            with tf.variable_scope('maxout_{}'.format(i)):
                components.append(self._linear(inputs, output_dim))
        outputs = tf.stack(components, 2)
        outputs = tf.reduce_max(outputs, -1)
        return outputs

    def gaussian_noise(self, inputs, std=0.001):
        noise = tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=std, dtype=tf.float32) 
        return inputs + noise

    def feedforward_attention(self, inputs, scope='feedforward_attention'):
        att_vector = tf.contrib.layers.fully_connected(
            scope=scope,
            inputs=inputs,
            weights_initializer=tf.initializers.truncated_normal(stddev=0.1),
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
            biases_initializer=None,
            num_outputs=1,
            activation_fn=None)
        att_vector = tf.exp(att_vector)/ tf.reduce_sum(tf.exp(att_vector))
        inputs = inputs*att_vector
        return tf.reduce_mean(inputs, axis=1)
