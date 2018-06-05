""" Prediction models.
"""

import typing as t

import numpy as np
import tensorflow as tf

import embed
import data
import op
from nn.base import Model, SoftmaxCrossEntropyMixin
from util.log import exec_log as log
from util.debug import *


class DIIN(SoftmaxCrossEntropyMixin, Model):
    
    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            scale_l1: float = 0.0,
            scale_l2: float = 0.0,
            ) -> None:
        super(DIIN, self).__init__()
        self._class_num = class_num
        self.scale_l1 = scale_l1
        self.scale_l2 = scale_l2

        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        op_kwargs = {'scale_l1': self.scale_l1,
                     'scale_l2': self.scale_l2,
                     'keep_prob': self.keep_prob
                    }

        with tf.variable_scope('embed') as s:
            # Word pretrained embeddings (300D)
            word_embed = tf.constant(embeddings.get_embeddings(),
                    dtype=tf.float32,
                    name='word_embed')
            x1_word_embed = tf.gather(word_embed, self.x1)
            x2_word_embed = tf.gather(word_embed, self.x2)

            # Character convolutional embeddings (100D)
            #char_embed = _get_variable('char_embed',
            #        shape=(256, 100),
            #        dtype=tf.float32)
            #def get_char_embed(x_char):
            #    embed = tf.gather(char_embed, x_char)
                # shape: [batch, seq_len, word_len, embed_dim]

            # Tag one-hot embeddings (72D)
            def get_tag_embed(x_ids, x_tags, x_len):
                x_tags *= tf.sequence_mask(x_len, tf.shape(x_tags)[1],
                                           dtype=tf.int32)
                # shape: [batch, seq_len]
                tag_embed = tf.one_hot(x_tags, data.SNLI.TAGS,
                                       dtype=tf.float32,
                                       name='char_embed')
                return tag_embed[:,:tf.shape(x_ids)[1]]
            x1_tag_embed = get_tag_embed(self.x1, self.tag1, self.len1)
            x2_tag_embed = get_tag_embed(self.x2, self.tag2, self.len2)

            # Merge embeddings
            #x1 = tf.concat([x1_word_embed, x1_tag_embed], 2)
            #x2 = tf.concat([x2_word_embed, x2_tag_embed], 2)
            x1 = x1_word_embed
            x2 = x2_word_embed

        with tf.variable_scope('encode') as s:
            def encode_fn(x):
                x = op.highway(x, scope='hw-1', **op_kwargs)
                x = op.highway(x, scope='hw-2', **op_kwargs)
                return x
            x1, x2 = map(encode_fn, [x1, x2])

        with tf.variable_scope('attent') as s:
            def forward(x):
                t = op.linear(x, scope='linear-1', **op_kwargs)
                t = op.linear(t, scope='linear-2', **op_kwargs)
                return t
            sim = tf.matmul(forward(x1), tf.matrix_transpose(forward(x2)))
            #sim = tf.matmul(op.linear(x1, **op_kwargs),
            #                tf.matrix_transpose(op.linear(x2, **op_kwargs)))
            alpha = tf.matmul(tf.nn.softmax(tf.matrix_transpose(sim)), x1)
            beta  = tf.matmul(tf.nn.softmax(sim), x2)

        with tf.variable_scope('compare') as s:
            x1 = op.linear(tf.concat([x1, beta ], 2), **op_kwargs)
            x2 = op.linear(tf.concat([x2, alpha], 2), **op_kwargs)

        #with tf.variable_scope('decode') as s:
        #    def decode_fn(x):
        #        x = op.highway(x, scope='hw-1', **op_kwargs)
        #        x = op.highway(x, scope='hw-2', **op_kwargs)
        #        return x
        #    v1, v2 = map(decode_fn, [x1, x2])

        with tf.variable_scope('aggregate') as s:
            # CHANGE
            #def reduce_mean(x, x_len):
            #    return (tf.reduce_sum(x, axis=1) /
            #            tf.expand_dims(tf.cast(x_len, tf.float32), -1))
            v = tf.concat([
                    #reduce_mean(v1, self.len1),
                    #reduce_mean(v2, self.len2),
                    #tf.reduce_max(v1, axis=1),
                    #tf.reduce_max(v2, axis=1),
                    tf.reduce_sum(x1, axis=1),
                    tf.reduce_sum(x2, axis=1)
                    ], 1)
            v = op.linear(v, scope='linear-1')
            v = op.linear(v, scope='linear-2')
            y_hat = op.linear(v, dim=self._class_num, **op_kwargs)

        self.evaluate_and_loss(y_hat)
