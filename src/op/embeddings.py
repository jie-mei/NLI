# _*_ coding: utf-8 _*_

import numpy as np
import tensorflow as tf

from typing import Union

def embedding(
        inputs: tf.Tensor,
        word_embeddings: np.ndarray,
        mask_paddings: bool = True,
        normalize: bool = False,
        trainable: bool = False,
        ):
    """ Embedding lookup for a sequence of integer IDs.

    Arguments:
        inputs: A sequence of word IDs.
        word_embeddings: Pretrained word embeddings.
        scope: The layer variable scope.
        trainable: Whether the word embedding is trainable.

    Inputs:
        2-D Tensor [batch, seq_len]

    Returns:
        3-D Tensor [batch, seq_len, embed_dim]
    """
    with tf.device('gpu:0'):
        embeddings = tf.constant(word_embeddings, tf.float32, name='embeddings')
        if normalize:
            embeddings = tf.nn.l2_normalize(embeddings, 1)
        if trainable:
            # TODO: Aviod reassign
            embeddings = tf.get_variable('embeddings', initializer=embeddings)
        return tf.gather(embeddings, inputs)
