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


class DIIN(SoftmaxCrossEntropyMixin, Model):
    
    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            scale_l1: float = 0.0,
            scale_l2: float = 0.000001,
            encode_dim: int = 300,
            fact_intr_dim: int = 10,
            fact_proj_dim: int = -1,
            char_filer_width: int = 5,
            char_embed_dim: int = 8,
            char_conv_dim: int = 100,
            lstm_unit: int = 300,
            ) -> None:
        super(DIIN, self).__init__()
        self._class_num = class_num
        self.scale_l1 = scale_l1
        self.scale_l2 = scale_l2
        self.encode_dim = encode_dim
        self.fact_proj_dim = fact_proj_dim
        self.fact_intr_dim = fact_intr_dim
        self.char_filter_width = char_filer_width
        self.char_embed_dim = char_embed_dim
        self.char_conv_dim = char_conv_dim
        self.lstm_unit = lstm_unit

        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        batch_size = tf.shape(self.x1)[0]
        padded_len1 = tf.shape(self.x1)[1]
        padded_len2 = tf.shape(self.x2)[1]

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
            # Highway encoding 
            def highway_encode(x):
                x = op.highway(x, scope='hw-1', dim=self.encode_dim, **op_kwargs)
                x = op.highway(x, scope='hw-2', dim=self.encode_dim, **op_kwargs)
                return x
            x1, x2 = map(highway_encode, [x1, x2])
            # shape: [batch, seq_len, encode_dim]

            # Self-attention
            def self_attent(x, padded_len):
                t1 = tf.tile(tf.expand_dims(x, 2), [1, 1, tf.shape(x)[1], 1])
                t2 = tf.tile(tf.expand_dims(x, 1), [1, tf.shape(x)[1], 1, 1])
                # shape: [batch, seq_len, seq_len, encode_dim]
                t = tf.reshape(tf.concat([t1, t2, t1 * t2], 3),
                        [batch_size, padded_len ** 2, 3 * self.encode_dim])
                # shape: [batch, seq_len^2, 3 * encode_dim]
                att = op.linear(t, dim=1, bias=None, activation_fn=None)
                # shape: [batch, seq_len^2, 1]
                att = tf.reshape(att, [batch_size, padded_len, padded_len])
                # shape: [batch, seq_len, seq_len]
                soft_align = tf.einsum('bik,bkj->bij', tf.nn.softmax(att), x)
                return op.gated_fuse(x, soft_align)
                # shape: [batch, seq_len, encode_dim]
            x1, x2 = map(lambda x, l: self_attent(x, l),
                    *zip((x1, padded_len1), (x2, padded_len2)))

        with tf.variable_scope('interact') as s:
            inter = tf.expand_dims(x1, 2) * tf.expand_dims(x2, 1)
            # shape: [batch, seq_len1, seq_len2, encode_dim]

        with tf.variable_scope('extract') as s:
            # Dense Net
            feats = op.conv2d(inter, self.encode_dim * 0.3, 1)
            # shape: [batch, seq_len1, seq_len2, encode_dim]
            feats = self.dense_block(feats, 'dense-block-1')
            feats = self.dense_trans(feats, 'dense-trans-1')
            feats = self.dense_block(feats, 'dense-block-2')
            feats = self.dense_trans(feats, 'dense-trans-2')
            feats = self.dense_block(feats, 'dense-block-3')
            feats = self.dense_trans(feats, 'dense-trans-3')

            shape = tf.shape(feats)
            feats = tf.reshape(feats, [shape[0], shape[1] * shape[2] * shape[3]])

        self.evaluate_and_loss(feats)

    def dense_block(self,
            feats,
            scope,
            num_layers: int = 8,
            growth_rate: int = 20,
            kernel_size: int = 3):
        with tf.variable_scope(scope):
            for i in range(num_layers):
                new_feats = op.conv2d(feats, growth_rate,
                        (kernel_size, kernel_size), scope='conv2d-%d' % i)
                feats = tf.concat([feats, new_feats], 3)
        return feats

    def dense_trans(self,
            feats,
            scope,
            transition_rate: float = 0.5,):
        with tf.variable_scope(scope):
            out_dim = int(int(feats.shape[-1]) * transition_rate)
            feats = op.conv2d(feats, out_dim, 1, activation_fn=None)
            feats = tf.nn.max_pool(feats, [1,2,2,1], [1,2,2,1], 'VALID')
        return feats
