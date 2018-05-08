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
from util.log import exec_log as log


def _make_dataset(
        dataset: data.Dataset,
        batch_size: int,
        bucket_boundaries: t.List[int] = [],
        shuffle: bool = True,
        pad: bool = True,
        shuffle_buffer_size: int = 40960,
        prefetch_buffer_size: int = -1,
        repeat_num: int = 1,
        seed: int = None
        ) -> t.Tuple[tf.data.Iterator, tf.Tensor]:
    """ Prepare a `tf.data.Dataset` for evaluation.

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
    dset = dset.cache()

    if shuffle and shuffle_buffer_size > 1:
        if tf.__version__ >= '1.6':
            dset = dset.apply(tf.contrib.data.shuffle_and_repeat(
                buffer_size=shuffle_buffer_size,
                count=repeat_num,
                seed=seed))
        else:
            dset = (dset.shuffle(shuffle_buffer_size, seed=seed)
                        .repeat(repeat_num))
    else:
        dset = dset.repeat(repeat_num)

    # Pack records with similar lengthes as batch.
    if pad:
        if bucket_boundaries:
            if tf.__version__ >= '1.8':
                log.debug('Generate batches using '
                          'tf.contrib.data.bucket_by_sequence_length')
                dset = dset.apply(tf.contrib.data.bucket_by_sequence_length(
                        lambda x1, x2, y, len1, len2: tf.maximum(len1, len2),
                        bucket_boundaries,
                        [batch_size] * (len(bucket_boundaries) + 1)))
            else:
                log.debug('Generate batches using tf.contrib.data.group_by_window')
                def bucketing(x1, x2, y, len1, len2):
                    size = tf.maximum(len1, len2)
                    bucket = tf.case(
                            [(size < b, lambda: i + 1)
                                    for i, b in enumerate(bucket_boundaries)],
                            default=lambda: 0,
                            exclusive=True)
                    return tf.to_int64(bucket)
                dset = dset.apply(tf.contrib.data.group_by_window(
                        key_func=bucketing,
                        reduce_func=lambda _, data: data.padded_batch(batch_size,
                                padded_shapes = ([None], [None], [], [], [])),
                        window_size=batch_size))
        else:
            log.debug('Generate padded batches without bucketing')
            dset = dset.padded_batch(batch_size,
                                     padded_shapes = ([None], [None], [], [], []))
    else:
        log.debug('Generate batches without padding input sequences')
        dset = dset.batch(batch_size)

    if prefetch_buffer_size <= 0:
        prefetch_buffer_size = 64 * batch_size
    return dset.prefetch(buffer_size=prefetch_buffer_size)


def _make_dataset_iterator(
        session: tf.Session,
        type_name: str,
        **args):
    dataset = _make_dataset(**args)
    iterator = getattr(dataset, 'make_' + type_name)()
    handle = session.run(iterator.string_handle())
    return iterator, handle


def _make_model_summary(model: nn.Model=None):
    # Summary setup
    tf.summary.scalar("Loss", model.loss)  # type: ignore
    tf.summary.scalar("Accuracy", model.performance)  # type: ignore
    # Plot all the parameters in tensorboard
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        tf.summary.histogram(name=v.name.replace(':','_'), values=v)
    return tf.summary.merge_all()


def _make_optimizer(type_name: str, **kwargs):
    kwargs['name'] = "optimizer"
    log.debug('Model optimzation using %s' % type_name)
    if type_name == 'AdamOptimizer' and kwargs['learning_rate'] != 0.001:
        log.warning('Apply learning rate %f with AdamOptimizer' %
                    kwargs['learning_rate'])
    return getattr(tf.train, type_name)(**kwargs)


def _make_config():
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    return config


def _iterate_dataset(session, model, iterator, handle, summary_writer, step):
    """ Run dataset in the given session and record the accuracy. """
    session.run(iterator.initializer)
    y_preds, y_trues = [], []  # mypy: ignore
    while True:
        try:
            true, pred = session.run([model.y, model.prediction],
                    feed_dict={model.handle: handle,
                               model.keep_prob: 1.0})  # disable dropout
            y_preds += pred,
            y_trues += true,
        except tf.errors.OutOfRangeError:
            break
    acc = sklearn.metrics.accuracy_score(
            np.concatenate(y_trues).tolist(),
            np.concatenate(y_preds).tolist())
    summary = tf.Summary(value=[
            tf.Summary.Value(tag='Accuracy', simple_value=acc)])
    summary_writer.add_summary(summary, step)


def _profile_and_exit(session, model, optimizer, handle):
    """ Profile the run metadata at the first iteration and exit the program.
    """
    from tensorflow.python.client import timeline
    run_metadata = tf.RunMetadata()
    for i in range(5):
        session.run([optimizer],
                feed_dict={model.handle: handle},
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_%d.json' % i, 'w') as f:
            f.write(ctf)
    log.info('Profiles are created! Now exit.')
    exit()


def _save_model(session, path, step):
    save_path = (tf.train.Saver(max_to_keep=100).save(session,
            build.get_save_path(path),
            global_step=step))
    return save_path


def train(name: str,
          batch_size: int = 256,
          epoch_num: int = 200,
          keep_prob: float = 0.8,
          learning_rate: float = 0.05,
          optimizer_type: str = 'AdagradOptimizer',
          model_type: str = 'Decomposable',
          data_name: str = 'SNLI',
          data_embedding: str = 'GloVe',
          data_pad: bool = True,
          record_every: int = 1000,
          validate_every: int = 10000,
          save_every: int = 100000,
          profiling: bool = False,
          seed: int = None,
          **kwargs
          ) -> None:

    # Data preparation
    model_path = build.get_model_path(name)
    shutil.rmtree(model_path, ignore_errors=True)  # remove previous trained

    # Network setup
    model = getattr(nn, model_type)(
            embeddings=data.load_embeddings(data_name, data_embedding),
            **kwargs)
    log.info(str(model))
    log.debug('Model parameters:\n\n\t' +
              '\n\t'.join(graph.print_trainable_variables().split('\n')))

    train_summary = _make_model_summary(model)

    # Control randomization
    if seed:
        log.info('Set random seed for data shuffling and graph computation: %d' % seed)
        tf.set_random_seed(seed)

    # Optimization
    optim = _make_optimizer(optimizer_type, learning_rate=learning_rate).minimize(model.loss)

    with tf.Session(config=_make_config()) as sess:
        sess.run(tf.global_variables_initializer())

        train_wtr = tf.summary.FileWriter(os.path.join(model_path, 'train'), sess.graph)
        valid_wtr = tf.summary.FileWriter(os.path.join(model_path, 'valid'))
        test_wtr = tf.summary.FileWriter(os.path.join(model_path, 'test'))

        train_iter, train_hd = _make_dataset_iterator(
                type_name='one_shot_iterator',
                dataset=data.load_dataset(data_name, 'train', data_embedding),
                batch_size=batch_size,
                bucket_boundaries=[20, 50],
                pad=data_pad,
                repeat_num=epoch_num,
                seed=seed,
                session=sess)
        valid_iter, valid_hd = _make_dataset_iterator(
                type_name='initializable_iterator',
                dataset=data.load_dataset(data_name, 'validation', data_embedding),
                batch_size=batch_size,
                shuffle=False,
                pad=data_pad,
                session=sess)
        test_iter, test_hd = _make_dataset_iterator(
                type_name='initializable_iterator',
                dataset=data.load_dataset(data_name, 'test', data_embedding),
                batch_size=batch_size,
                shuffle=False,
                pad=data_pad,
                session=sess)

        if profiling:
            _profile_and_exit(sess, model, optim, train_hd)

        step = 1
        pbar = tqdm.tqdm(total=save_every, desc='Train', unit='batch')
        try:
            while True:
                if step % record_every == 0:
                    summary, _, loss = sess.run(
                            [train_summary, optim, model.loss],
                            feed_dict={model.handle: train_hd,
                                       model.keep_prob: keep_prob})
                    pbar.set_postfix(loss='{:.3f}'.format(loss))
                    train_wtr.add_summary(summary, step)
                else:
                    sess.run([optim], feed_dict={model.handle: train_hd,
                                                 model.keep_prob: keep_prob})

                if step % validate_every == 0:
                    pbar.set_description('Valid')
                    _iterate_dataset(sess, model, valid_iter, valid_hd, valid_wtr, step)
                    pbar.set_description('Test')
                    _iterate_dataset(sess, model, test_iter, test_hd, test_wtr, step)
                    pbar.set_description('Train')

                if step % save_every == 0:
                    save_path = _save_model(sess, model_path, step)
                    pbar.set_description(save_path)
                    pbar.update(1)
                    pbar.close()
                    pbar = tqdm.tqdm(total=save_every, desc='Train', unit='batch')
                else:
                    pbar.update(1)

                step += 1

        except tf.errors.OutOfRangeError:
            save_path = _save_model(sess, model_path, step)
            pbar.set_description(save_path)
            log.info('Training finished!')


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
    log.debug('Input arguments:\n\n\t%s\n' %
              '\n\t'.join('%-20s %s' % (k + ':', v) for k, v in kwargs.items()))
    train(**kwargs) # type: ignore
