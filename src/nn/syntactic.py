import typing as t

import numpy as np
import tensorflow as tf

import embed
from nn.attentive import AttentiveModel
from util.debug import *


class Syntactic(AttentiveModel):

    def post_project(self, x1: tf.Tensor, x2: tf.Tensor):
        x1 = self.unfold_tree(x1, self.temp1, self.tag1, self.len1, 'x1')
        x2 = self.unfold_tree(x2, self.temp2, self.tag2, self.len2, 'x2')
        return x1, x2


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

            # Pad a leading 0 to align with the unfolded tree
            tag = tf.pad(tag, [[0, 0], [1, 0]])
            tag = tf_Print(tag, [tag], message='tag=')
            #tag = tf_Print(tag, [temp], message='temp=')
            # shape: [batch, 1 + tag_len]
            # NOTE: tag_len <= seq_len + temp_len

            # find the next available position (zero embedding)
            top = tf.expand_dims(len_ + 1, -1)
            # shape: [batch, 1]

            i = tf.constant(0)
            def loop_cond(i, tree, temp, tag, batch_size):
                return tf.less(i, tf.shape(temp)[1])
            def loop_body(i, tree, temp, tag, batch_size):
                c_idx = tf.gather(temp, i, axis=1)
                #c_idx = tf_Print(c_idx, [i, c_idx], message='c_idx=', first_n=3)
                # shape: [batch, temp_size, 2]
                p_idx = tf.concat(
                        [tf.expand_dims(tf.range(batch_size), -1), top + i],
                        axis=1)
                # shape: [batch, 2]
                p_tag = tf.gather_nd(tag, p_idx)
                # shape: [batch]
                c_embed = tf.gather_nd(tree, c_idx)
                # shape: [batch, temp_size, embed_dim]
                c_tag = tf.gather_nd(tag, c_idx)
                c_tag = tf_Print(c_tag, [i, tf.shape(tag), c_tag], message='c_tag=', first_n=3)
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


class SyntacticForward(Syntactic):

    def merge_fn(self,
            c_embeds: tf.Tensor,  # 3D: [batch, temp_size, embed_dim]
            c_tags: tf.Tensor,    # 2D: [batch, temp_size]
            p_tags: tf.Tensor     # 1D: [batch]
            )-> tf.Tensor:        # 2D: [batch, embed_dim]
        if not hasattr(self, 'input_dim'):
            c_shape = c_embeds.get_shape()
            self.input_dim = c_shape[1] * c_shape[2]
        return self.forward(tf.reshape(c_embeds, [-1, self.input_dim]))


class SyntacticForwardAtt(Syntactic):

    def __init__(self, tags_num: int = 72, **kwargs)-> None:
        self.tags_num = tags_num
        super(SyntacticForwardAtt, self).__init__(**kwargs)

    def merge_fn(self,
            c_embeds: tf.Tensor,  # 3D: [batch, temp_size, embed_dim]
            c_tags: tf.Tensor,    # 2D: [batch, temp_size]
            p_tags: tf.Tensor     # 1D: [batch]
            )-> tf.Tensor:        # 2D: [batch, embed_dim]
        if not hasattr(self, 'embed_dim'):
            self.embed_dim = c_embeds.get_shape()[-1]
        with tf.variable_scope('merge', reuse=tf.AUTO_REUSE):
            tag_weights = tf.get_variable('tag_weight',
                    shape=(self.tags_num, self.embed_dim),
                    dtype=tf.float32,
                    #initializer=tf.initializers.constant(1))
                    initializer=tf.initializers.truncated_normal(1))
            # shape: [tags, embed_dim]

            # Forward attention with POS tag
            e_t = self.forward(c_embeds)
            # shape: [batch, temp_size, embed_dim]
            t_t = tf.gather(tag_weights, c_tags)
            # shape: [batch, temp_size, embed_dim]
            att = tf.nn.softmax(e_t * t_t, axis=1)
            # shape: [batch, temp_size, embed_dim]
            return tf.reduce_sum(att * c_embeds, axis=1)
