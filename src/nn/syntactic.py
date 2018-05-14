import typing as t

import numpy as np
import tensorflow as tf

import embed
from nn.decomposable import Decomposable


class Syntactic(Decomposable):

    def post_project(self, x1: tf.Tensor, x2: tf.Tensor):
        x1 = self.unfold_tree(x1, self.temp1, self.tag1, self.len1, 'x1')
        x2 = self.unfold_tree(x2, self.temp2, self.tag2, self.len2, 'x2')
        return x1, x2


    def unfold_tree(self,
            embed: tf.Tensor,  # 3D: [batch, seq_len, embed_dim]
            temp: tf.Tensor,   # 3D: [batch, tmp_len, tmp_size]
            tag: tf.Tensor,    # 2D: [batch, seq_len + tmp_len]
            len_: tf.Tensor,   # 1D: [batch]
            suffix: str):
        with tf.name_scope('unfold_tree_%s' % suffix):
            batch_size = tf.shape(embed)[0]

            # Create a container of size (x.len + temp.len + 1) for the
            # unfoldered tree embeddings, where one zero embedding
            # vector padded at head.
            tree = tf.pad(embed, [[0, 0], [1, tf.shape(temp)[1]], [0, 0]])
            # NOTE: This is a trick to have a fixed embedding dimension in the
            # construction time. This is used for initializing variables (e.g.
            # in a linear transofrmation layer).
            tree = tf.reshape(tree, [batch_size, -1, embed.get_shape()[-1]])
            # shape: [batch, 1 + seq_len + tmp_len, embed]

            # Add batch index to each template position.
            temp = tf.expand_dims(temp, -1)
            bidx = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1, 1]),
                        [1, tf.shape(temp)[1], tf.shape(temp)[2], 1])
            temp = tf.concat([bidx, temp], axis=3)
            # shape: [batch, tmp_len, tmp_size, 2]

            # Pad a leading 0 to align with the unfolded tree
            tag = tf.pad(tag, [[0, 0], [1, 0]])
            # shape: [batch, 1 + seq_len + tmp_len]

            # find the next available position (zero embedding)
            top = tf.expand_dims(len_ + 1, -1)
            # shape: [batch, 1]

            i = tf.constant(0)
            def loop_cond(i, tree):
                return tf.less(i, tf.shape(temp)[1])
            def loop_body(i, tree):
                c_idx = tf.gather(temp, i, axis=1)
                # shape: [batch, tmp_size, 2]
                p_idx = tf.concat(
                        [tf.expand_dims(tf.range(batch_size), -1), top + i],
                        axis=1)
                #import pdb; pdb.set_trace()
                # shape: [batch, 2]
                p_tag = tf.gather_nd(tag, p_idx)
                # shape: [batch]
                c_embed = tf.gather_nd(tree, c_idx)
                # shape: [batch, tmp_size, embed]
                c_tag = tf.gather_nd(tag, c_idx)
                # shape: [batch, tmp_size]
                p_embed = self.merge_fn(c_embed, c_tag, p_tag)
                tree += tf.scatter_nd(
                        indices=p_idx,
                        updates=p_embed,
                        shape=tf.shape(tree))
                i += 1
                return i, tree
            i_loop, x_loop = tf.while_loop(loop_cond, loop_body, [i, tree],
                    #shape_invariants=[[1], list(tf.shape(tree))],
                    parallel_iterations=1)
            return x_loop


    def merge_fn(self,
            c_embeds: tf.Tensor,  # 3D: [batch, tmp_size, embed]
            c_tags: tf.Tensor,    # 2D: [batch, tmp_size]
            p_tags: tf.Tensor     # 1D: [batch]
            )-> tf.Tensor:        # 2D: [batch, embed]
        return tf.reduce_mean(c_embeds, axis=1)
