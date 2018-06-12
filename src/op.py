import typing as t

import tensorflow as tf


DEFAULT_VAR_INIT = tf.variance_scaling_initializer()
DEFAULT_SCALE_L1 = 0.0
DEFAULT_SCALE_L2 = 0.0


def get_variable(
        name: str,
        initializer: t.Callable = DEFAULT_VAR_INIT,
        scale_l1: float = DEFAULT_SCALE_L1,
        scale_l2: float = DEFAULT_SCALE_L2,
        dtype: tf.DType = tf.float32,
        **kwargs):
    """ Get an existing tf.Variable with these parameters or create a new one.

    This function calls `tf.get_variable` while overriding some default
    parameters. All ops using variables MUST use this function for consistency.
    """
    return tf.get_variable(name,
            initializer=initializer,
            regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1, scale_l2),
            dtype=dtype,
            **kwargs)


def conv2d(
        inputs,
        num_outputs,
        kernel_size,
        padding='SAME',
        weight_initializer: t.Callable = DEFAULT_VAR_INIT,
        biases_initializer: t.Callable = DEFAULT_VAR_INIT,
        scale_l1: float = DEFAULT_SCALE_L1,
        scale_l2: float = DEFAULT_SCALE_L2,
        reuse=tf.AUTO_REUSE,
        activation_fn=tf.nn.relu,
        scope='conv2d'):
    """ Conduct an N-D convolution (1 <= N <=3).

    This function calls `tf.contrib.layers.conv2d` while overriding some default
    parameters in order to be consistant with `get_variable`.
    """
    return tf.contrib.layers.conv2d(inputs,
            num_outputs,
            kernel_size,
            padding=padding,
            weights_initializer=weight_initializer,
            weights_regularizer=tf.contrib.layers.l1_l2_regularizer(
                scale_l1, scale_l2),
            biases_initializer=biases_initializer,
            biases_regularizer=tf.contrib.layers.l1_l2_regularizer(
                scale_l1, scale_l2),
            reuse=reuse,
            activation_fn=activation_fn,
            scope=scope)


def linear(
        inputs: tf.Tensor,
        dim: int = -1,
        activation_fn: t.Callable = tf.nn.relu,
        bias: bool = True,
        drop_after: bool = True,
        keep_prob: float = 1.0,
        scope: t.Union[str, tf.VariableScope] = None,
        reuse: bool = tf.AUTO_REUSE,
        **kwargs):
    """
    Inputs:  3D-Tensor [batch, seq_len, input_dim], or
             2D-Tensor [batch, input_dim]
    Returns: 3D-Tensor [batch, seq_len, dim], or
             2D-Tensor [batch, dim]
    """
    dim = dim if dim > 0 else int(inputs.shape[-1])
    t = tf.nn.dropout(inputs, keep_prob) if not drop_after else inputs
    with tf.variable_scope(scope if scope else 'linear', reuse=reuse):
        t_shape = tf.shape(t)
        w = get_variable('weight', shape=[t.get_shape()[-1], dim], **kwargs)
        output_rank = len(t.get_shape())
        if output_rank == 3:
            t = tf.reshape(t, [-1, t.shape[2]])
        t = tf.matmul(t, w)
        if bias:
            b = get_variable('bias', shape=[dim], **kwargs)
            t += b
        t = activation_fn(t) if activation_fn else t
        if output_rank == 3:
            t = tf.reshape(t, [-1, t_shape[1], dim])
    t = tf.nn.dropout(t, keep_prob) if drop_after else t
    return t


def highway(
        inputs: tf.Tensor,
        scope: t.Union[str, tf.VariableScope] = None,
        reuse: bool = tf.AUTO_REUSE,
        **kwargs):
    """
    Inputs:  3D-Tensor [batch, seq_len, input_dim], or
             2D-Tensor [batch, input_dim]
    Returns: 3D-Tensor [batch, seq_len, dim], or
             2D-Tensor [batch, dim]
    """
    with tf.variable_scope(scope if scope else 'highway', reuse=reuse):
        trans = linear(inputs, scope='trans', **kwargs)
        gate = linear(inputs, scope='gate', activation_fn=tf.sigmoid, **kwargs)
        if inputs.shape[-1] != gate.shape[-1]:
            # Additional afine transformation to project input into the shape
            inputs = linear(inputs, scope='affine', **kwargs)
        return gate * trans + (1 - gate) * inputs


def gated_fuse(
        t1: tf.Tensor,
        t2: tf.Tensor,
        scope: t.Union[str, tf.VariableScope] = None,
        reuse: bool = tf.AUTO_REUSE,
        **kwargs
        ):
    """
    Inputs:  3D-Tensor [batch, seq_len, t1_dim]
    Returns: 3D-Tensor [batch, seq_len, t1_dim]
    """
    with tf.variable_scope(scope if scope else 'gated-fuse', reuse=reuse):
        proj = lambda inputs, scope: linear(inputs,
                dim=t1.shape[-1],
                scope=scope,
                activation_fn=None,
                **kwargs)
        # NOTE: two tensors are concated in the paper
        with tf.variable_scope('trans'):
            trans = tf.nn.relu(proj(t1, 'linear1') + proj(t2, 'linear2'))
        with tf.variable_scope('gate'):
            gate  = tf.sigmoid(proj(t1, 'linear1') + proj(t2, 'linear2'))
        return gate * trans + (1 - gate) * t1


def lstm(
        inputs: tf.Tensor,
        seq_len: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = 'gru',
        dynamic: bool = False,
        bidirectional: bool = False,
        scope: t.Union[str, tf.VariableScope] = None,
        reuse: bool = tf.AUTO_REUSE
        ):
    # type and shape check
    if not isinstance(seq_len, int):
        if len(seq_len.get_shape()) == 2:
            seq_len = tf.squeeze(seq_len, axis=1)
        if seq_len.dtype.is_floating:
            seq_len = tf.cast(seq_len, tf.int32)
        if len(inputs.get_shape()) == 4:
            inputs = tf.squeeze(inputs, axis=3)
    with tf.variable_scope(scope if scope else cell_type, reuse=reuse):
        with tf.name_scope('multilayer-rnn'):
            cell = _get_multi_rnn_cell(hidden_size,
                    num_layers=num_layers,
                    cell_type=cell_type,
                    bidirectional=bidirectional)
            rnn_outputs = _get_rnn_outputs(cell, inputs, seq_len,
                    dynamic=dynamic,
                    bidirectional=bidirectional)
        return rnn_outputs


def _get_multi_rnn_cell(hidden_size, num_layers=1, cell_type='gru',
        bidirectional=False):
    cell_types = {'lstm': tf.nn.rnn_cell.LSTMCell,
                  'gru': tf.nn.rnn_cell.GRUCell}
    cell = cell_types.get(cell_type, None)

    rnn_params = {'num_units': hidden_size,
                  'activation': tf.nn.relu}
    if cell_type == 'lstm':
        rnn_params['use_peepholes'] = True

    cell_fw = tf.contrib.rnn.MultiRNNCell(
            [cell(**rnn_params) for _ in range(num_layers)])
    if bidirectional is False:
        return cell_fw
    else:
        cell_bw = tf.contrib.rnn.MultiRNNCell(
                        [cell(**rnn_params) for _ in range(num_layers)])
        return cell_fw, cell_bw


def _get_rnn_outputs(cell, inputs, seq_len, dynamic=True, bidirectional=False):
    if dynamic is True:
        if bidirectional is True:
            (cell_fw, cell_bw) = cell
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, tf.cast(inputs, tf.float32),
                    sequence_length=seq_len,
                    time_major=False,
                    dtype='float32')
            output_fw, output_bw = outputs
            outputs = tf.concat(axis=2, values=[output_fw, output_bw])
        else:
            outputs, state = tf.nn.dynamic_rnn(cell, tf.cast(inputs, tf.float32),
                sequence_length=seq_len,
                time_major=False,
                dtype='float32')
    else:
        inputs = tf.unstack(inputs, axis=1)
        inputs = [tf.cast(i, tf.float32) for i in inputs]
        outputs, output_state_fw, output_state_bw = \
                tf.nn.static_bidirectional_rnn(
                    cell_fw, cell_bw, inputs,
                    sequence_length=seq_len,
                    dtype='float32')
        outputs = tf.stack(outputs, axis=1)
    return outputs
