import tensorflow as tf


def lstm(inputs, seq_len, hidden_size, num_layers=1, cell_type='gru',
        bidirectional=False):
    # type and shape check
    if not isinstance(seq_len, int):
        if len(seq_len.get_shape()) == 2:
            seq_len = tf.squeeze(seq_len, axis=1)
        if seq_len.dtype.is_floating:
            seq_len = tf.cast(seq_len, tf.int32)
    if len(inputs.get_shape()) == 4:
        inputs = tf.squeeze(inputs, axis=3)

    with tf.name_scope('multilayer-rnn'):
        cell = _get_multi_rnn_cell(hidden_size, num_layers=num_layers,
                cell_type=cell_type, bidirectional=bidirectional)
        rnn_outputs = _get_rnn_outputs(cell, inputs, seq_len,
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
                    cell_fw,
                    cell_bw,
                    tf.cast(inputs, tf.float32),
                    sequence_length=seq_len,
                    time_major=False,
                    dtype='float32')
            output_fw, output_bw = outputs
            outputs = tf.concat(axis=2, values=[output_fw, output_bw])
        else:
            outputs, state = tf.nn.dynamic_rnn(
                    cell,
                    tf.cast(inputs, tf.float32),
                    sequence_length=seq_len,
                    time_major=False,
                    dtype='float32')
    else:
        inputs = tf.unstack(inputs, axis=1)
        inputs = [tf.cast(i, tf.float32) for i in inputs]
        (outputs, output_state_fw, output_state_bw) = tf.nn.static_bidirectional_rnn(
                cell_fw,
                cell_bw,
                inputs,
                sequence_length=seq_len,
                dtype='float32')
        outputs = tf.stack(outputs, axis=1)
    return outputs
