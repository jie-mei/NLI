""" Prediction models.
"""

import typing as t

import numpy as np
import tensorflow as tf

import embed
import data
import op
import nn
from nn.base import Model, WeightedSoftmaxCrossEntropyMixin
from util.log import exec_log as log
from util.debug import *


class ESIM(WeightedSoftmaxCrossEntropyMixin, Model):
    
    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            scale_l1: float = 0.0,
            scale_l2: float = 0.0,
            lstm_unit: int = 300,
            seq_len: int = 0,
            char_filer_width: int = 5,
            char_embed_dim: int = 8,
            char_conv_dim: int = 100,
            class_weights: t.List[float] = [1.1, 1, 1],
            ) -> None:
        super(ESIM, self).__init__()
        self._class_num = class_num
        self.class_weights = class_weights
        self.scale_l1 = scale_l1
        self.scale_l2 = scale_l2
        self.lstm_unit = lstm_unit
        self.seq_len = seq_len
        self.char_filter_width = char_filer_width
        self.char_embed_dim = char_embed_dim
        self.char_conv_dim = char_conv_dim

        op_kwargs = {'scale_l1': self.scale_l1,
                     'scale_l2': self.scale_l2,
                     'keep_prob': self.keep_prob,
                     'drop_after': False}

        with tf.variable_scope('embed') as s:
            def set_seq_len(x):
                x_len = tf.shape(x)[1]
                return tf.cond(
                        tf.less(self.seq_len, x_len),
                        lambda: x[:,:self.seq_len],
                        lambda: tf.pad(x, [[0, 0], [0, self.seq_len - x_len]]))
            if self.seq_len > 0:
                x1, x2 = map(set_seq_len, [self.x1, self.x2])
            else:
                x1, x2 = self.x1, self.x2

            #embed_init_var = embeddings.get_embeddings()
            #embed = op.get_variable('embeddings',
            #        shape=embed_init_var.shape,
            #        initializer=tf.constant_initializer(embed_init_var))

            #embed = tf.constant(embeddings.get_embeddings(),
            #                    dtype=tf.float32,
            #                    name='embeddings')
            #x1, x2 = map(lambda x: tf.gather(embed, x), [x1, x2])

            # Word pretrained embeddings (300D)
            word_embed = tf.constant(embeddings.get_embeddings(),
                    dtype=tf.float32,
                    name='word_embed')
            word_embed1, word_embed2 = map(lambda x: tf.gather(word_embed, x),
                    [self.x1, self.x2])
            embed_dim = word_embed.get_shape()[-1]

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
            #x1 = tf.concat([word_embed1, char_embed1, tag_embed1], 2)
            #x2 = tf.concat([word_embed2, char_embed2, tag_embed2], 2)
            x1 = tf.concat([word_embed1, char_embed1], 2)
            x2 = tf.concat([word_embed2, char_embed2], 2)

            x1 = self.unfold_tree(x1, self.temp1, self.tag1, self.len1, 'x1')
            x2 = self.unfold_tree(x2, self.temp2, self.tag2, self.len2, 'x2')

        with tf.variable_scope('encode', reuse=tf.AUTO_REUSE) as s:
            x1, x2 = map(lambda x: tf.nn.dropout(x, self.keep_prob), [x1, x2])
            #import pdb; pdb.set_trace()
            x1, x2 = map(self.bilstm, [x1, x2])
            # shape: [batch, seq_len, embed_dim * 2]

        with tf.variable_scope('attent') as s:
            sim = tf.matmul(x1, tf.matrix_transpose(x2))
            alpha = tf.matmul(tf.nn.softmax(tf.matrix_transpose(sim)), x1)
            beta  = tf.matmul(tf.nn.softmax(sim), x2)
            x1 = tf.concat([x1, beta,  x1 * beta,  x1 - beta ], 2)
            x2 = tf.concat([x2, alpha, x2 * alpha, x2 - alpha], 2)
            # shape: [batch, seq_len, embed_dim * 8]

        with tf.variable_scope('decode', reuse=tf.AUTO_REUSE) as s:
            x1, x2 = map(lambda x: op.linear(x, dim=embed_dim, **op_kwargs),
                         [x1, x2])
            # NOTE: dropout here in the author's code
            # shape: [batch, seq_len, embed_dim]
            x1, x2 = map(self.bilstm, [x1, x2])
            # shape: [batch, seq_len, embed_dim * 2]

        with tf.variable_scope('aggregate') as s:
            def pool(x):
                return tf.concat([
                         tf.reduce_sum(x, axis=1),
                         tf.reduce_max(x, axis=1)
                     ], 1)
            y_hat = op.linear(tf.concat([pool(x1), pool(x2)], 1),
                    dim=embed_dim, 
                    activation_fn=tf.nn.tanh,
                    scope='linear-1',
                    **op_kwargs)
            # shape: [batch, embed_dim * 8]
            y_hat = op.linear(y_hat,
                    dim=self._class_num, 
                    activation_fn=None,
                    scope='linear-2',
                    **op_kwargs)
            # shape: [batch, class_num]

        self.evaluate_and_loss(y_hat, self.class_weights)

    def bilstm(self, x):
        # shape: [batch, seq_len, embed_dim]
        if self.seq_len > 0:
            # Static RNN
            #lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, self.lstm_unit,
            #        direction='bidirectional')
            #return tf.transpose(lstm(tf.transpose(x, [1, 0, 2]))[0], [1, 0, 2])
            x_seq = tf.unstack(
                    tf.reshape(x, [-1, self.seq_len, x.get_shape()[-1]]),
                    axis=1)
            out, _, _ = tf.nn.static_bidirectional_rnn(
                            cell_fw=tf.contrib.rnn.LSTMBlockCell(self.lstm_unit),
                            cell_bw=tf.contrib.rnn.LSTMBlockCell(self.lstm_unit),
                            inputs=x_seq,
                            dtype=tf.float32)
            return tf.stack(out, axis=1)
        else:
            # Dynamic RNN
            (outputs_fw, outputs_bw), (states_fw, states_bw) = \
                    tf.nn.bidirectional_dynamic_rnn(
                            cell_fw=tf.contrib.rnn.LSTMBlockCell(self.lstm_unit),
                            cell_bw=tf.contrib.rnn.LSTMBlockCell(self.lstm_unit),
                            inputs=x,
                            dtype=tf.float32)
            return tf.concat([outputs_fw, outputs_bw], 2)
            # shape: [batch, seq_len, embed_dim * 2]

    def unfold_tree(self,
            embed: tf.Tensor,  # 3D: [batch, seq_len, embed_dim]
            temp: tf.Tensor,   # 3D: [batch, temp_len, temp_size]
            tag: tf.Tensor,    # 2D: [batch, seq_len + temp_len]
            len_: tf.Tensor,   # 1D: [batch]
            suffix: str):
        with tf.name_scope('unfold_tree_%s' % suffix):
            batch_size = tf.shape(embed)[0]

            # Create a container of size (x.len + temp.len + 1) for the
            # unfoldered tree embeddings, where one zero embedding
            # vector is padded at head.
            tree = tf.pad(embed, [[0, 0], [1, tf.shape(temp)[1]], [0, 0]])
            # NOTE: This is a trick to have a fixed embedding dimension in the
            # construction time. This is used for initializing variables (e.g.
            # in a linear transofrmation layer).
            tree = tf.reshape(tree, [batch_size, -1, embed.get_shape()[-1]])
            # shape: [batch, 1 + seq_len + temp_len, embed_dim]

            # Add batch index to each template position.
            temp = tf.expand_dims(temp, -1)
            bidx = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1, 1]),
                        [1, tf.shape(temp)[1], tf.shape(temp)[2], 1])
            temp = tf.concat([bidx, temp], axis=3)
            # shape: [batch, temp_len, temp_size, 2]
            temp = tf.cast(temp, tf.float32) # NOTE: register tf.gather in GPU.

            # Pad a leading 0 to align with the unfolded tree
            tag = tf.pad(tag, [[0, 0], [1, 0]])
            tag = tf.cast(tag, tf.float32) # NOTE: register tf.gather in GPU.
            # shape: [batch, 1 + tag_len]
            # NOTE: tag_len <= seq_len + temp_len

            # find the next available position (zero embedding)
            top = tf.expand_dims(len_ + 1, -1)
            # shape: [batch, 1]

            i = tf.constant(1)
            def loop_cond(i, tree, temp, tag, batch_size):
                return tf.less(i, tf.shape(temp)[1])
            def loop_body(i, tree, temp, tag, batch_size):
                c_idx = tf.gather(temp, i, axis=1)
                c_idx = tf.cast(c_idx, tf.int32)  # NOTE: restore type
                # shape: [batch, temp_size, 2]
                p_idx = tf.concat(
                        [tf.expand_dims(tf.range(batch_size), -1), top + i],
                        axis=1)
                # shape: [batch, 2]
                p_tag = tf.gather_nd(tag, p_idx)
                p_tag = tf.cast(p_tag, tf.int32)  # NOTE: restore type
                # shape: [batch]
                c_embed = tf.gather_nd(tree, c_idx)
                # shape: [batch, temp_size, embed_dim]
                c_tag = tf.gather_nd(tag, c_idx)
                c_tag = tf.cast(c_tag, tf.int32)  # NOTE: restore type
                # shape: [batch, temp_size]
                p_embed = self.merge_fn(c_embed, c_tag, p_tag)
                tree += tf.scatter_nd(
                        indices=p_idx,
                        updates=p_embed,
                        shape=tf.shape(tree))
                i += 1
                return [i, tree, temp, tag, batch_size]
            _, x_loop, _, _, _ = tf.while_loop(loop_cond, loop_body,
                    [i, tree, temp, tag, batch_size],
                    parallel_iterations=1)
            return x_loop

    def merge_fn(self,
            c_embeds: tf.Tensor,  # 3D: [batch, temp_size, embed_dim]
            c_tags: tf.Tensor,    # 2D: [batch, temp_size]
            p_tags: tf.Tensor     # 1D: [batch]
            )-> tf.Tensor:        # 2D: [batch, embed_dim]
        return tf.reduce_mean(c_embeds, axis=1)
