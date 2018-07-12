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


class ESIM(SoftmaxCrossEntropyMixin, Model):
    
    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            scale_l1: float = 0.0,
            scale_l2: float = 0.0,
            lstm_unit: int = 300,
            seq_len: int = 0,
            ) -> None:
        super(ESIM, self).__init__()
        self._class_num = class_num
        self.scale_l1 = scale_l1
        self.scale_l2 = scale_l2
        self.lstm_unit = lstm_unit
        self.seq_len = seq_len

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
                #x1 = tf.Print(x1, [x1[:,0]], 'x1')
                #x2 = tf.Print(x2, [x2[:,0]], 'x2')
            else:
                x1, x2 = self.x1, self.x2

            #embed_init_var = embeddings.get_embeddings()
            #embed = op.get_variable('embeddings',
            #        shape=embed_init_var.shape,
            #        initializer=tf.constant_initializer(embed_init_var))
            embed = tf.constant(embeddings.get_embeddings(),
                                dtype=tf.float32,
                                name='embeddings')
            embed_dim = embed.get_shape()[-1]
            x1, x2 = map(lambda x: tf.gather(embed, x), [x1, x2])

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
                    activation_fn=tf.nn.tanh,
                    scope='linear-1',
                    **op_kwargs)
            # shape: [batch, embed_dim * 8]
            y_hat = op.linear(y_hat, dim=self._class_num, 
                    activation_fn=None,
                    scope='linear-2',
                    **op_kwargs)
            # shape: [batch, class_num]

        self.evaluate_and_loss(y_hat)

    def bilstm(self, x):
        # shape: [batch, seq_len, embed_dim]
        if self.seq_len > 0:
            # Static RNN
            #lstm_layer = tf.contrib.cudnn_rnn.CudnnLSTM(1, self.lstm_unit,
            #        input_mode='linear_input',
            #        direction='bidirectional')
            #return lstm_layer(x, training=self.is_training.eval())
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
                            #cell_fw=tf.nn.rnn_cell.LSTMCell(self.lstm_unit),
                            #cell_bw=tf.nn.rnn_cell.LSTMCell(self.lstm_unit),
                            inputs=x,
                            dtype=tf.float32)
            return tf.concat([outputs_fw, outputs_bw], 2)
            # shape: [batch, seq_len, embed_dim * 2]
