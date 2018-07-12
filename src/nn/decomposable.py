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


class DecomposableMod(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            project_dim: int = 200,
            intra_attention: bool = False,
            bias_init: float = 0,
            char_filer_width: int = 5,
            char_embed_dim: int = 8,
            char_conv_dim: int = 100,
            lstm_unit: int = 300,
            ) -> None:
        super(DecomposableMod, self).__init__()
        self._class_num = class_num
        self.project_dim = project_dim
        self.intra_attention = intra_attention
        self.bias_init = bias_init
        self.lstm_unit = lstm_unit
        self.char_filter_width = char_filer_width
        self.char_embed_dim = char_embed_dim
        self.char_conv_dim = char_conv_dim

        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        with tf.variable_scope('embed', reuse=tf.AUTO_REUSE) as s:
            # Word pretrained embeddings (300D)
            word_embed = tf.constant(embeddings.get_embeddings(),
                    dtype=tf.float32,
                    name='word_embed')
            word_embed1, word_embed2 = map(lambda x: tf.gather(word_embed, x),
                    [self.x1, self.x2])

            # Tag one-hot embeddings (72D)
            #def embed_tags(x_ids, x_tags):
            #    tag_weight = op.get_variable('tag_weight',
            #                                 shape=(data.SNLI.TAGS, 1))
            #    x_tags = x_tags[:,:tf.shape(x_ids)[1]]
            #    return tf.gather(tag_weight, x_tags)
            #    # shape: [batch, seq_len, 1]
            #tag_embed1, tag_embed2 = map(embed_tags,
            #        *zip((self.x1, self.tag1), (self.x2, self.tag2)))
            def embed_tags(x_ids, x_tags, x_len):
                x_tags *= tf.sequence_mask(x_len, tf.shape(x_tags)[1],
                                           dtype=tf.int32)
                # shape: [batch, seq_len]
                tag_embed = tf.one_hot(x_tags, data.SNLI.TAGS,
                                       dtype=tf.float32,
                                       name='tag_embed')
                return tag_embed[:,:tf.shape(x_ids)[1]]
            tag_embed1, tag_embed2 = map(embed_tags,
                    *zip((self.x1, self.tag1, self.len1),
                         (self.x2, self.tag2, self.len2)))

            # Merge embeddings
            #x1 = tf.concat([word_embed1, char_embed1, tag_embed1], 2)
            #x2 = tf.concat([word_embed2, char_embed2, tag_embed2], 2)
            x1 = tf.concat([word_embed1, tag_embed1], 2)
            x2 = tf.concat([word_embed2, tag_embed2], 2)
            #x1 = word_embed1 * tag_embed1
            #x2 = word_embed2 * tag_embed2

            #import pdb; pdb.set_trace()
            x1, x2 = self.intra(x1, x2) if intra_attention else (x1, x2)

        with tf.variable_scope('encode', reuse=tf.AUTO_REUSE) as s:
            def lstm_encode(x):
                # shape: [batch, seq_len, embed_dim]
                (outputs_fw, outputs_bw), (states_fw, states_bw) = \
                        tf.nn.bidirectional_dynamic_rnn(
                                cell_fw=tf.nn.rnn_cell.LSTMCell(self.lstm_unit),
                                cell_bw=tf.nn.rnn_cell.LSTMCell(self.lstm_unit),
                                inputs=x,
                                dtype=tf.float32)
                outputs = tf.concat([outputs_fw, outputs_bw], 2)
                return tf.nn.dropout(outputs, self.keep_prob)
                # shape: [batch, seq_len, embed_dim * 2]
            x1, x2 = map(lstm_encode, [x1, x2])

        with tf.variable_scope('attent') as s:
            #sim = self.attention(x1, x2)
            sim = tf.matmul(x1, tf.matrix_transpose(x2))
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
