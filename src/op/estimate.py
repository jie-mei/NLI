# _*_ coding: utf-8 _*_

import numpy as np
import tensorflow as tf

from typing import Union

def mean_sequare_error(
        inputs: tf.Tensor,
        y: tf.Tensor,
        scope: Union[str, tf.VariableScope] = tf.get_variable_scope(),
        name: str = 'estimate',
        ):
    """ Make prediction and compute loss according to the mean sequare error.

    Arguments:
        inputs: A sequence of word IDs.
        y: The ground truth lables.
        scope: The layer variable scope.
        name: The layer name.

    Inputs:
        1-D Tensor [batch]: inputs
        1-D Tensor [batch]: y

    Returns:
        1-D Tensor [batch]: prediction
        1-D Tensor [batch]: loss
    """
    with tf.variable_scope(scope, default_name=name):
        with tf.name_scope('predition'):
            pred = tf.cast(inputs > 0.5, tf.int32)
        with tf.name_scope('loss'):
            mse_loss = tf.losses.mean_squared_error(
                    tf.cast(y, tf.float32), inputs)
            reg_loss = tf.reduce_sum(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES))
            loss = tf.add(mse_loss, reg_loss)
    return pred, loss
