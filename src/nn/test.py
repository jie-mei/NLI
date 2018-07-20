import typing as t

import numpy as np
import tensorflow as tf

import embed
import op
import data
from nn.base import Model, SoftmaxCrossEntropyMixin
from util.log import exec_log as log


class Test(SoftmaxCrossEntropyMixin, Model):

    def __init__(self,
            embeddings: embed.IndexedWordEmbedding,
            class_num: int,
            scale_l1: float = 0.0,
            scale_l2: float = 0.0,
            lstm_unit: int = 300,
            ) -> None:
        super(Test, self).__init__()
        self._class_num = class_num
        self.scale_l1 = scale_l1
        self.scale_l2 = scale_l2
        self.lstm_unit = lstm_unit

        self.batch_size = tf.shape(self.x1)[0]
        self.embed_dim = embeddings.get_embeddings().shape[-1]
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        self.op_var_kwargs = {'scale_l1': self.scale_l1,
                              'scale_l2': self.scale_l2}
        self.op_kwargs = {'scale_l1': self.scale_l1,
                          'scale_l2': self.scale_l2,
                          'keep_prob': self.keep_prob,
                          'drop_after': False}

        with tf.variable_scope('embed') as s:
            #embed_init_var = embeddings.get_embeddings()
            #embed = op.get_variable('embeddings',
            #        shape=embed_init_var.shape,
            #        initializer=tf.constant_initializer(embed_init_var))
            embed = tf.constant(embeddings.get_embeddings(),
                                dtype=tf.float32,
                                name='embeddings')
            embed_dim = embed.get_shape()[-1]
            x1, x2 = map(lambda x: tf.gather(embed, x), [self.x1, self.x2])

        with tf.variable_scope('unfold', reuse=tf.AUTO_REUSE) as s:
            x1 = self.unfold_tree(x1, self.temp1, self.tag1, self.len1, 'x1')
            x2 = self.unfold_tree(x2, self.temp2, self.tag2, self.len2, 'x2')

        with tf.variable_scope('encode', reuse=tf.AUTO_REUSE) as s:
            x1, x2 = map(lambda x: tf.nn.dropout(x, self.keep_prob), [x1, x2])
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
            x1, x2 = map(lambda x: op.linear(x, embed_dim, **self.op_kwargs),
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
                    **self.op_kwargs)
            # shape: [batch, embed_dim * 8]
            y_hat = op.linear(y_hat, dim=self._class_num, 
                    activation_fn=None,
                    scope='linear-2',
                    **self.op_kwargs)
            # shape: [batch, class_num]

        self.evaluate_and_loss(y_hat)

    def bilstm(self, x):
        # shape: [batch, seq_len, embed_dim]
        (outputs_fw, outputs_bw), (states_fw, states_bw) = \
                tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=tf.nn.rnn_cell.LSTMCell(self.lstm_unit),
                        cell_bw=tf.nn.rnn_cell.LSTMCell(self.lstm_unit),
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
                #c_tag = tf.Print(c_tag, [i, c_idx, c_tag, p_tag], message='i, c_idx, c_tag, p_tag', first_n=20, summarize=10)
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
        temp_size = data.SNLI.TEMP_DROP_VAL
        fact_dim = 10

        tag_c_wght = op.get_variable('tag_child_weights',
                shape=[data.SNLI.TAGS, self.embed_dim * fact_dim],
                **self.op_var_kwargs)
        c_wght = tf.gather(tag_c_wght, c_tags)
        # shape: [batch, temp_size, embed_dim * fact_dim]
        c_wght = tf.reshape(c_wght, [
                self.batch_size, temp_size * self.embed_dim, fact_dim])
        # shape: [batch, temp_size * embed_dim, fact_dim]

        #tag_c_bias = op.get_variable('tag_child_biases',
        #        shape=[data.SNLI.TAGS, fact_dim],
        #        **self.op_var_kwargs)
        #c_bias = tf.gather(tag_c_bias, c_tags)
        # shape: [batch, temp_size, fact_dim]
        #c_bias = tf.reshape(c_bias, [self.batch_size, temp_size * fact_dim])
        # shape: [batch, temp_size, fact_dim]

        tag_p_wght = op.get_variable('tag_parent_weights',
                shape=[data.SNLI.TAGS, self.embed_dim * fact_dim],
                **self.op_var_kwargs)
        p_wght = tf.gather(tag_p_wght, p_tags)
        # shape: [batch, embed_dim * fact_dim]
        p_wght = tf.reshape(p_wght, [self.batch_size, self.embed_dim, fact_dim])
        # shape: [batch, embed_dim, fact_dim]

        #tag_p_bias = op.get_variable('tag_parent_biases',
        #        shape=[data.SNLI.TAGS, fact_dim, 1],
        #        **self.op_var_kwargs)
        #c_bias = tf.gather(tag_p_bias, c_tags)
        # shape: [batch, fact_dim, 1]

        w = tf.einsum('bik,bjk->bij', c_wght, p_wght)
        # shape: [batch, temp_size * embed_dim, embed_dim]
        #b = tf.einsum('bik,bjk->bij', c_wght, p_wght)
        # shape: [batch, embed_dim, 1]

        x = tf.reshape(c_embeds, [self.batch_size, temp_size * self.embed_dim, 1])
        # shape: [batch, temp_size * embed_dim, 1]

        y = tf.einsum('bki,bkj->bij', w, x)
        # shape: [batch, embed_dim, 1]

        return tf.squeeze(y, [-1])
