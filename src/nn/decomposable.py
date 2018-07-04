import typing as t

import numpy as np
import tensorflow as tf

import embed
import op
import data
from nn.base import Model, SoftmaxCrossEntropyMixin
from util.log import exec_log as log


class Decomposable(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            project_dim: int = 200,
            intra_attention: bool = False,
            bias_init: float = 0,
            ) -> None:
        super(Decomposable, self).__init__()
        self.project_dim = project_dim
        self.intra_attention = intra_attention
        self.bias_init = bias_init
        self._class_num = class_num

        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        with tf.variable_scope('embed') as s:
            embed = tf.constant(embeddings.get_embeddings(),
                                dtype=tf.float32,
                                name='embeddings')
            x1, x2 = map(lambda x: tf.gather(embed, x), [self.x1, self.x2])
            project = lambda x: self.linear(x, self.project_dim, bias=False)
            x1, x2 = map(project, [x1, x2])
            x1, x2 = self.intra(x1, x2) if intra_attention else (x1, x2)

        with tf.variable_scope('attent') as s:
            sim = self.attention(x1, x2)
            self.attent_map = sim  # for visualization
            alpha = tf.matmul(tf.nn.softmax(tf.matrix_transpose(sim)), x1)
            beta  = tf.matmul(tf.nn.softmax(sim), x2)
        
        with tf.variable_scope('compare') as s:
            v1 = self.forward(tf.concat([x1, beta ], 2))
            v2 = self.forward(tf.concat([x2, alpha], 2))

        with tf.variable_scope('aggregate') as s:
            v1 = tf.reduce_sum(v1, axis=1)
            v2 = tf.reduce_sum(v2, axis=1)
            y_hat = self.forward(tf.concat([v1, v2], 1))
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
        op_kwargs = {
                'keep_prob': self.keep_prob,
                'activation_fn': tf.nn.relu
                }
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            t = op.linear(inputs,
                    dim=self.project_dim,
                    scope='linear-1',
                    **op_kwargs)
            t = op.linear(t, scope='linear-2', **op_kwargs)
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


class DecomposableWithCharEmbed(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            project_dim: int = 200,
            intra_attention: bool = False,
            bias_init: float = 0,
            char_filer_width: int = 5,
            char_embed_dim: int = 8,
            char_conv_dim: int = 100,
            ) -> None:
        super(DecomposableWithCharEmbed, self).__init__()
        self._class_num = class_num
        self.project_dim = project_dim
        self.intra_attention = intra_attention
        self.bias_init = bias_init
        self.char_filter_width = char_filer_width
        self.char_embed_dim = char_embed_dim
        self.char_conv_dim = char_conv_dim

        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        with tf.variable_scope('embed') as s:
            # Word pretrained embeddings (300D)
            word_embed = tf.constant(embeddings.get_embeddings(),
                    dtype=tf.float32,
                    name='word_embed')
            def embed(x):
                x = tf.gather(word_embed, x)
                # shape: [batch, seq_len, embed_len]
                att = tf.expand_dims(x, 2) * tf.expand_dims(x, 3)
                return x
            word_embed1, word_embed2 = map(lambda x: tf.gather(word_embed, x),
                    [self.x1, self.x2])

            # Character convolutional embeddings (`char_conv_dim`D)
            char_embed = op.get_variable('char_embed',
                    shape=(256, char_embed_dim))
            char_filter = op.get_variable('char_filter',
                    shape=(1, self.char_filter_width, self.char_embed_dim,
                            self.char_conv_dim))
            def embed_chars(x_char):
                embed = tf.gather(char_embed, x_char)
                # shape: [batch, seq_len, word_len, embed_dim]
                conv = tf.nn.conv2d(embed, char_filter, [1, 1, 1, 1], 'VALID')
                # shape: [batch, seq_len, word_len - filter_width + 1, conv_dim]
                return tf.reduce_max(conv, 2)
                # shape: [batch, seq_len, conv_dim]
            char_embed1, char_embed2 = map(embed_chars, [self.char1, self.char2])

            # Tag one-hot embeddings (72D)
            def embed_tags(x_ids, x_tags, x_len):
                x_tags *= tf.sequence_mask(x_len, tf.shape(x_tags)[1],
                                           dtype=tf.int32)
                # shape: [batch, seq_len]
                tag_embed = tf.one_hot(x_tags, data.SNLI.TAGS,
                                       dtype=tf.float32,
                                       name='char_embed')
                return tag_embed[:,:tf.shape(x_ids)[1]]
            tag_embed1, tag_embed2 = map(embed_tags,
                    *zip((self.x1, self.tag1, self.len1),
                         (self.x2, self.tag2, self.len2)))

            # Merge embeddings
            x1 = tf.concat([word_embed1, char_embed1, tag_embed1], 2)
            x2 = tf.concat([word_embed2, char_embed2, tag_embed2], 2)

            #import pdb; pdb.set_trace()
            x1, x2 = self.intra(x1, x2) if intra_attention else (x1, x2)

        with tf.variable_scope('attent') as s:
            sim = self.attention(x1, x2)
            alpha = tf.matmul(tf.nn.softmax(tf.matrix_transpose(sim)), x1)
            beta  = tf.matmul(tf.nn.softmax(sim), x2)
        
        with tf.variable_scope('compare') as s:
            v1 = self.forward(tf.concat([x1, beta ], 2))
            v2 = self.forward(tf.concat([x2, alpha], 2))

        with tf.variable_scope('aggregate') as s:
            v1 = tf.reduce_sum(v1, axis=1)
            v2 = tf.reduce_sum(v2, axis=1)
            y_hat = self.forward(tf.concat([v1, v2], 1))
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
        op_kwargs = {
                'keep_prob': self.keep_prob,
                'activation_fn': tf.nn.relu
                }
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            t = op.linear(inputs,
                    dim=self.project_dim,
                    scope='linear-1',
                    **op_kwargs)
            t = op.linear(t, scope='linear-2', **op_kwargs)
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
