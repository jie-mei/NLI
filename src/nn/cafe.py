""" Prediction models.
"""

import typing as t

import numpy as np
import tensorflow as tf

import embed
import data
import op
import nn
from nn.base import Model, SoftmaxCrossEntropyMixin
from util.log import exec_log as log
from util.debug import *


class CAFE(SoftmaxCrossEntropyMixin, Model):
    
    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            scale_l1: float = 0.0,
            scale_l2: float = 0.000001,
            encode_dim: int = 300,
            fact_dim: int = 10,
            char_filer_width: int = 5,
            char_embed_dim: int = 8,
            char_conv_dim: int = 100,
            lstm_unit: int = 300,
            ) -> None:
        super(CAFE, self).__init__()
        self._class_num = class_num
        self.scale_l1 = scale_l1
        self.scale_l2 = scale_l2
        self.encode_dim = encode_dim
        self.fact_dim = fact_dim
        self.char_filter_width = char_filer_width
        self.char_embed_dim = char_embed_dim
        self.char_conv_dim = char_conv_dim
        self.lstm_unit = lstm_unit

        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        op_kwargs = {'scale_l1': self.scale_l1,
                     'scale_l2': self.scale_l2,
                     'keep_prob': self.keep_prob}

        with tf.variable_scope('embed') as s:
            # Word pretrained embeddings (300D)
            word_embed = tf.constant(embeddings.get_embeddings(),
                    dtype=tf.float32,
                    name='word_embed')
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

        with tf.variable_scope('encode') as s:
            def encode(x):
                x = op.highway(x, scope='hw-1', dim=self.encode_dim, **op_kwargs)
                x = op.highway(x, scope='hw-2', dim=self.encode_dim, **op_kwargs)
                return x
            x1, x2 = map(encode, [x1, x2])
            # shape: [batch, seq_len, encode_dim]

        with tf.variable_scope('attent') as s:
            # Alignment
            def co_attent(t1, t2):
                t1 = op.linear(t1, **op_kwargs)
                t2 = op.linear(t2, **op_kwargs)
                return tf.matmul(t1, tf.matrix_transpose(t2))
                # shape: [batch, seq_len1, seq_len2]
            with tf.variable_scope('inter-align') as s:
                att = co_attent(x1, x2)
                inter1 = tf.matmul(tf.nn.softmax(att), x2)
                inter2 = tf.matmul(tf.nn.softmax(tf.matrix_transpose(att)), x1)
            with tf.variable_scope('intra-align') as s:
                def self_attent(x):
                    att = co_attent(x, x)
                    return x * tf.reduce_sum(att, 2, keep_dims=True)
                intra1, intra2 = map(self_attent, [x1, x2])
            def align_fact(x, x_align, scope):
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    f1 = self.fact('fact-concat', tf.concat([x, x_align], 2))
                    f2 = self.fact('fact-sub',    x - x_align)
                    f3 = self.fact('fact-mul',    x * x_align)
                    return tf.stack([f1, f2, f3], 2)
                # shape: [batch, seq_len, 3]

            # TODO: variables may not be shared between different facts
            x1 = tf.concat([x1,
                            align_fact(x1, inter1, 'inter'),
                            align_fact(x1, intra1, 'intra')], 2)
            x2 = tf.concat([x2,
                            align_fact(x2, inter2, 'inter'),
                            align_fact(x2, intra2, 'intra')], 2)

        with tf.variable_scope('sequence', reuse=tf.AUTO_REUSE) as s:
            def lstm_encode(x):
                # shape: [batch, seq_len, encode_dim + 6]
                outputs, states = tf.nn.dynamic_rnn(
                        cell=tf.nn.rnn_cell.LSTMCell(self.lstm_unit),
                        inputs=x,
                        dtype=tf.float32)
                return outputs
            x1, x2 = map(lstm_encode, [x1, x2])

        with tf.variable_scope('pooling') as s:
            def pool(x):
                return tf.concat([
                        tf.reduce_max(x, axis=1),
                        tf.reduce_sum(x, axis=1),
                        ], 1)
                # shape: [batch, dim]
            x1, x2 = map(pool, [x1, x2])

        with tf.variable_scope('decode') as s:
            x = tf.concat([x1, x2, x1 - x2, x1 * x2], 1)
            x = op.highway(x, scope='hw-1', dim=self.encode_dim, **op_kwargs)
            x = op.highway(x, scope='hw-2', dim=self.encode_dim, **op_kwargs)
            y_hat = op.linear(x, dim=self._class_num, activation_fn=None)

        self.evaluate_and_loss(y_hat)

    def fact(self, scope, x):
        """ Factorize input vector into a feature scalar.

        Input:
            3D-Tensor: [batch, seq_len, input_dim]
        Output:
            2D-Tensor: [batch, seq_len]
        """
        return self.fact_impl2(scope, x)

    # Factoriztion
    def fact_impl1(self, scope, x):
        input_dim = x.get_shape()[2]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            fact_wght = op.get_variable('fact_weight',
                    shape=(input_dim))
            fact_bias = op.get_variable('fact_bias', shape=(1))
            fact_intr = op.get_variable('fact_inter',
                    shape=(input_dim, self.fact_dim))
        l = (tf.reduce_sum(x * tf.reshape(fact_wght, [1, 1, -1]), -1)
                + fact_bias)
        # shape: [batch, seq_len]
        intr_mat = tf.matmul(fact_intr, tf.matrix_transpose(fact_intr))
        # shape: [input, input_dim]
        mask = tf.sequence_mask(tf.range(input_dim),
                maxlen=input_dim,
                dtype=tf.float32)
        # shape: [encode_dim, input_dim]
        p = tf.reduce_sum(
                tf.matmul(tf.expand_dims(x, 2), tf.expand_dims(x, 3)) *
                # shape: [batch, seq_len, input_dim, input_dim]
                tf.expand_dims(tf.expand_dims(intr_mat, 0), 0),
                #tf.expand_dims(tf.expand_dims(intr_mat * mask, 0), 0),
                # shape: [1, 1, input_dim, input_dim]
                [2, 3])
        # shape: [batch, seq_len]
        return l + p

    # Factoriztion
    def fact_impl2(self, scope, x):
        input_dim = x.get_shape()[2]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            fact_wght = op.get_variable('fact_weight',
                    shape=(input_dim))
            fact_bias = op.get_variable('fact_bias', shape=(1))
            fact_intr = op.get_variable('fact_inter',
                    shape=(input_dim, self.fact_dim))
        l = (tf.reduce_sum(x * tf.reshape(fact_wght, [1, 1, -1]), -1)
                + fact_bias)
        # shape: [batch, seq_len]

        intr_mat = tf.matmul(fact_intr, tf.matrix_transpose(fact_intr))
        # shape: [input, input_dim]
        mask = tf.sequence_mask(tf.range(input_dim),
                maxlen=input_dim,
                dtype=tf.float32)
        # shape: [encode_dim, input_dim]

        i = tf.constant(0)
        x_shape = tf.shape(x)
        batch_size, seq_len = x_shape[0], x_shape[1]
        p = tf.reshape(tf.zeros([batch_size]), [batch_size, -1])
        def loop_cond(i, x, p, seq_len):
            return tf.less(i, seq_len)
        def loop_body(i, x, p, seq_len):
            x_vect = x[:,i]
            # shape: [batch, input_dim]
            #x_mat = tf.matmul(tf.expand_dims(x_vect, 1),
            #                  tf.expand_dims(x_vect, 2))
            # NOTE: Avoid Internal Error: Blas xGEMMBatched launch failed
            x_mat = (tf.tile(tf.expand_dims(x_vect, 1), [1, input_dim, 1]) *
                     tf.tile(tf.expand_dims(x_vect, 2), [1, 1, input_dim]))
            # shape: [batch, input_dim, input_dim]
            p_i = tf.reduce_sum(intr_mat * x_mat, [1, 2])
            p_i = tf.expand_dims(p_i, 1)
            # shape: [batch, 1]
            p = tf.concat([p, p_i], 1)
            return [i, x, p, seq_len]
        _, _, p_loop, _ = tf.while_loop(loop_cond, loop_body,
                [i, x, p, seq_len],
                parallel_iterations=1)
        return l + p_loop[:,1:]
