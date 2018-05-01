""" Model training. """

import os
import sys
import typing as t

import numpy as np
import shutil
import sklearn
import tensorflow as tf
import tqdm

import embed
import data
import nn
from util import build, parse, graph
from util.annotation import print_section
from util.log import exec_log as log


@print_section
def _print_trainable_variables():
    graph.print_trainable_variables()


@print_section
def _print_number_of_variables(model):
    print("Total Variables: %d" % model.count_parameters())


@print_section
def _print_model_setup(model):
    print(model)


def _prepare_dataset(
        dataset: data.Dataset,
        batch_size: int,
        bucket_boundaries: t.List[int] = [],
        shuffle: bool = True,
        shuffle_buffer_size: int = 40960,
        prefetch_buffer_size: int = -1,
        repeat_num: int = 1,
        ) -> tf.data.Dataset:
    """ Prepare tf.data.Dataset for evaluation.

    Args:
        dataset: A dataset.
        batch_size: The batch size.
        shuffle_buffer_size: The buffer size for random shuffling. Disable
            random shuffling when `shuffle_buffer_size` is smaller than or equal
            to 1.
        prefetch_buffer_size: The buffer size for prefetching. This parameter is
            used, when shuffling is permitted. When given a non-positive value,
            `prefetch_buffer_size` will adapt to `batch_size`.
        repeat_time: The number of times the records in the dataset are
            repeated.
    """
    dset = tf.data.Dataset.from_generator(
            lambda: ((x1, x2, y, len(x1), len(x2))
                     for x1, x2, y in
                     zip(dataset.x1_ids, dataset.x2_ids, dataset.labels)),
            output_types=(tf.int32,) * 5,
            output_shapes=(tf.TensorShape([None]),
                           tf.TensorShape([None]),
                           tf.TensorShape([]),
                           tf.TensorShape([]),
                           tf.TensorShape([])))

    if shuffle and shuffle_buffer_size > 1:
        if tf.__version__ >= '1.6':
            dset = dset.apply(tf.contrib.data.shuffle_and_repeat(
                shuffle_buffer_size, repeat_num))
        else:
            dset = dset.shuffle(shuffle_buffer_size).repeat(repeat_num)
    else:
        dset = dset.repeat(repeat_num)

    # Pack records with similar lengthes as batch.
    if tf.__version__ >= '1.8':
        log.debug('Generate batches using '
                  'tf.contrib.data.bucket_by_sequence_length')
        dset = dset.apply(tf.contrib.data.bucket_by_sequence_length(
                lambda x1, x2, y, len1, len2: tf.maximum(len1, len2),
                [20, 50],
                [batch_size] * 3))
    else:
        log.debug('Generate batches using tf.contrib.data.group_by_window')
        def bucketing(x1, x2, y, len1, len2):
            size = tf.maximum(len1, len2)
            bucket = tf.case([(size < 20, lambda: 1),
                              (size > 50, lambda: 2)],
                             default=lambda: 0,
                             exclusive=True)
            return tf.to_int64(bucket)
        dset = dset.apply(tf.contrib.data.group_by_window(
                key_func=bucketing,
                reduce_func=lambda _, data: data.padded_batch(batch_size,
                        padded_shapes = ([None], [None], [], [], [])),
                window_size=batch_size))

    if prefetch_buffer_size <= 0:
        prefetch_buffer_size = 32 * batch_size
    return dset.prefetch(buffer_size=prefetch_buffer_size)


def train(name: str,
          batch_size: int = 256,
          epoch_num: int = 200,
          learning_rate: float = 0.05,
          data_name: str = 'SNLI',
          data_embedding: str = 'GloVe',
          validate: bool = True,
          **kwargs
          ) -> None:

    # Data preparation
    model_path = build.get_model_path(name)
    shutil.rmtree(model_path, ignore_errors=True)  # remove previous trained

    train_data = _prepare_dataset(
            dataset=data.load_dataset(data_name, 'train', data_embedding),
            batch_size=batch_size,
            bucket_boundaries=[20, 50])
    valid_data = _prepare_dataset(
            dataset=data.load_dataset(data_name, 'validation', data_embedding),
            batch_size=batch_size,
            shuffle=False)
    test_data = _prepare_dataset(
            dataset=data.load_dataset(data_name, 'test', data_embedding),
            batch_size=batch_size,
            shuffle=False)

    # Network setup
    model = nn.Decomposeable(
            embeddings=data.load_embeddings(data_name, data_embedding),
            **kwargs)
    _print_model_setup(model)
    _print_trainable_variables()
    _print_number_of_variables(model)

    # Summary setup
    tf.summary.scalar("Loss", model.loss)
    tf.summary.scalar("Accuracy", model.performance)
    # Plot all the parameters in tensorboard
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        tf.summary.histogram(name=v.name.replace(':','_'), values=v)
    train_summary = tf.summary.merge_all()

    # Optimization
    optimizer = (tf.train.AdagradOptimizer(learning_rate, name="optimizer")
            .minimize(model.loss))
    #optimizer = (tf.train.AdamOptimizer(name="optimizer")
    #        .minimize(model.loss))

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        train_wtr = tf.summary.FileWriter(os.path.join(model_path, 'train'), sess.graph)
        valid_wtr = tf.summary.FileWriter(os.path.join(model_path, 'valid'))
        test_wtr = tf.summary.FileWriter(os.path.join(model_path, 'test'))

        sess.run(init)
        train_init_op = model.data_iterator.make_initializer(train_data)
        valid_init_op = model.data_iterator.make_initializer(valid_data)
        test_init_op = model.data_iterator.make_initializer(test_data)

        step = 0
        pbar = tqdm.tqdm(range(1, epoch_num + 1), desc='Train', unit='epoch')
        for e in pbar:

            # valid
            pbar.set_description('Valid')
            sess.run(valid_init_op)
            y_preds, y_trues = [], []  # type: t.List[int], t.List[int]
            while True:
                try:
                    true, pred = sess.run([model.y, model.prediction])
                    y_preds += pred,
                    y_trues += true,
                except tf.errors.OutOfRangeError:
                    break
            acc = sklearn.metrics.accuracy_score(
                    np.concatenate(y_trues).tolist(),
                    np.concatenate(y_preds).tolist())
            valid_wtr.add_summary(tf.Summary(value=[
                        tf.Summary.Value(tag='Accuracy', simple_value=acc)
                    ]), step)

            # test
            pbar.set_description('Test')
            sess.run(test_init_op)
            y_preds, y_trues = [], []  # mypy: ignore
            while True:
                try:
                    true, pred = sess.run([model.y, model.prediction])
                    y_preds += pred,
                    y_trues += true,
                except tf.errors.OutOfRangeError:
                    break
            acc = sklearn.metrics.accuracy_score(
                    np.concatenate(y_trues).tolist(),
                    np.concatenate(y_preds).tolist())
            test_wtr.add_summary(tf.Summary(value=[
                        tf.Summary.Value(tag='Accuracy', simple_value=acc)
                    ]), step)

            # Training
            pbar.set_description('Train')
            sess.run(train_init_op)
            while True:
                try:
                    if not step % 100:
                        summary, _, loss = sess.run(
                                [train_summary, optimizer, model.loss])
                        pbar.set_postfix(loss='{:.3f}'.format(loss))
                        train_wtr.add_summary(summary, step)
                    else:
                        sess.run([optimizer])
                    step += 1
                except tf.errors.OutOfRangeError:
                    break

        save_path = (tf.train.Saver(max_to_keep=100)
                .save(sess, build.get_save_path(model_path), global_step=e))
        print("model saved as", save_path)


if __name__ == "__main__":
    # Disable the debugging INFO and WARNING information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # Testing reads arguments from both file and commend line. Commend line
    # arguments can override ones parsed from file.
    kwargs = parse.parse_args(sys.argv)
    if 'file' in kwargs:
        # Use the file name as the default model name.
        fname = os.path.basename(kwargs['file'])  # type: ignore
        kwargs['name'] = fname[:fname.rfind('.')]  # type: ignore
        kwargs = {**parse.parse_yaml(kwargs['file'], mode='train'), **kwargs}
        del kwargs['file']
    train(**kwargs) # type: ignore
