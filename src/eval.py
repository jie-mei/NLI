#!/usr/bin/env python

""" Model evaluation. """

import glob
import os
import sys
import re
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
        cache: bool = False,
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
    output_shapes = ([None], [None], [], [], [],
                     [None, 4], [None, 4],
                     [None], [None]) # type: tuple
    dset = tf.data.Dataset.from_generator(
            lambda: ((x1, x2, y, len(x1), len(x2),
                            feat1[0], feat2[0],  # templates
                            feat1[1], feat2[1])  # tags
                     for x1, x2, y, feat1, feat2 in
                     zip(dataset.x1_ids,
                         dataset.x2_ids,
                         dataset.labels,
                         dataset.x1_feats,
                         dataset.x2_feats)),
            output_types=(tf.int32,) * 9,
            output_shapes=output_shapes)
    if cache:
        log.debug('Cache dataset during computation.')
        dset = dset.cache()
    else:
        log.debug('Do not cache dataset during computation.')

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
                        (lambda x1, x2, y, len1, len2, temp1, temp2, tag1, tag2:
                                tf.maximum(len1, len2)),
                        bucket_boundaries,
                        [batch_size] * (len(bucket_boundaries) + 1)))
            else:
                log.debug('Generate batches using '
                          'tf.contrib.data.group_by_window')
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
                                padded_shapes=output_shapes),
                        window_size=batch_size))
        else:
            log.debug('Generate padded batches without bucketing')
            dset = dset.padded_batch(batch_size, padded_shapes=output_shapes)
    else:
        log.debug('Generate batches without padding input sequences')
        dset = dset.batch(batch_size)

    if prefetch_buffer_size <= 0:
        prefetch_buffer_size = 64 * batch_size
    return dset.prefetch(buffer_size=prefetch_buffer_size)


def _make_dataset_iterator(
        session: tf.Session,
        type_name: str,
        handle_name: str,
        **args):
    dataset = _make_dataset(**args)
    iterator = getattr(dataset, 'make_' + type_name)()
    handle = session.run(iterator.string_handle(handle_name))
    return iterator, handle


def _make_model_summary(model: nn.Model=None):
    # Summary setup
    tf.summary.scalar("Loss", model.loss)  # type: ignore
    tf.summary.scalar("Accuracy", model.performance)  # type: ignore
    # Plot all the parameters in tensorboard
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        tf.summary.histogram(name=v.name.replace(':','_'), values=v)
    return tf.summary.merge_all()


def _search_var_list(var_regex_list: t.Union[t.List[str], str]):
    if isinstance(var_regex_list, str):
        var_regex_list = [var_regex_list]
    if var_regex_list:
        pattern_list = [re.compile(regex) for regex in var_regex_list]
        var_list = [var for var in tf.trainable_variables()
                    if any(map(lambda p: p.match(var.name), pattern_list))]
        log.info('Partical updation on parameters: \n\n\t%s\n' %
                 '\n\t'.join(map(lambda v: v.name, var_list)))
        return var_list


def _make_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #config.allow_soft_placement=True
    #config.log_device_placement=True
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
                feed_dict={model.handle: handle,
                           model.keep_prob: 1.0},
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_%d.json' % i, 'w') as f:
            f.write(ctf)
    log.info('Profiles are created! Now exit.')
    exit()


def _copy_checkpoint(from_model, to_path, step):
    """ Copy checkpoint files of a specific training step. """
    from_path = build.get_model_path(from_model)
    log.info('Copy step %d checkpoint files from: %s' % (step, from_path))
    files = glob.glob('{}/model-{}.*'.format(from_path, step))
    for f in files:
        shutil.copy(f, to_path)


def _save_model(session, path, step):
    save_path = (tf.train.Saver(max_to_keep=100).save(session,
            build.get_save_path(path),
            global_step=step))
    return save_path


def _restore_model(session, model_path, step):
    saved_path = build.get_saved_model(model_path, step)
    log.info('Restore pre-trained model from: %s' % saved_path)
    tf.train.Saver().restore(session, saved_path)


def train(name: str,
          model_type: str,
          batch_size: int = 256,
          epoch_num: int = 200,
          keep_prob: float = 0.8,
          train_regex_list: t.Union[t.List[str], str] = None,
          data_name: str = 'SNLI',
          data_embedding: str = 'GloVe',
          data_pad: bool = True,
          data_cache: bool = False,
          data_seed: int = None,
          record_every: int = 1000,
          validate_every: int = 10000,
          save_every: int = 100000,
          restore_from: str = None,
          restore_step: int = None,
          profiling: bool = False,
          seed: int = None,
          debug: bool = False,
          optimization_params: dict = {},
          **kwargs
          ) -> None:

    # Data preparation
    model_path = build.get_model_path(name)
    shutil.rmtree(model_path, ignore_errors=True)  # remove previous trained

    # Network setup
    model = getattr(nn, model_type)(
            embeddings=data.load_embeddings(data_name, data_embedding, data_seed),
            **kwargs)
    log.info(str(model))
    log.debug('Model parameters:\n\n\t' +
              '\n\t'.join(graph.print_trainable_variables().split('\n')))


    # Control randomization
    if seed:
        log.info('Set random seed for data shuffling and graph computation: %d' % seed)
        tf.set_random_seed(seed)

    'learning_rate global_step decay_steps decay_rate',
    # Setup learning rate
    global_step = tf.Variable(0, trainable=False, name='global_step')
    lr_manager = nn.optimization.LearningRateManager(
        learning_rates=optimization_params['learning_rates'],
        global_step=global_step,
        decay_steps=optimization_params.get('decay_steps', None),
        decay_rate=optimization_params.get('decay_rate', None))

    # Optimization
    optimizer_names = optimization_params.get('optimizer_names', None)
    if not optimizer_names:
        optimizer_names = [optimization_params.get('optimizer_type', 'AdagradOptimizer')]

    minimize_args = {'loss': model.loss,
        'var_list': _search_var_list(train_regex_list),
        'global_step': global_step}
    optim = nn.optimization.get_optimizer(
        optimizer_names,
        lr_manager,
        minimize_args)

    train_summary = _make_model_summary(model)

    with tf.Session(config=_make_config()) as sess:
        if debug:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        train_wtr = tf.summary.FileWriter(os.path.join(model_path, 'train'), sess.graph)
        valid_wtr = tf.summary.FileWriter(os.path.join(model_path, 'valid'))
        test_wtr = tf.summary.FileWriter(os.path.join(model_path, 'test'))

        dataset_opts = {
                'pad': data_pad,      
                'batch_size': batch_size,
                'session': sess,
                }
        train_iter, train_hd = _make_dataset_iterator(
                type_name='one_shot_iterator',
                handle_name='train_handle',
                dataset=data.load_dataset(
                        data_name, 'train', data_embedding, data_seed),
                bucket_boundaries=[20, 50],
                repeat_num=epoch_num,
                cache=data_cache,
                seed=seed,
                **dataset_opts)
        valid_iter, valid_hd = _make_dataset_iterator(
                type_name='initializable_iterator',
                handle_name='valid_handle',
                dataset=data.load_dataset(
                        data_name, 'validation', data_embedding, data_seed),
                shuffle=False,
                cache=True,
                **dataset_opts)
        test_iter, test_hd = _make_dataset_iterator(
                type_name='initializable_iterator',
                handle_name='test_handle',
                dataset=data.load_dataset(
                        data_name, 'test', data_embedding, data_seed),
                shuffle=False,
                cache=True,
                **dataset_opts)

        step = 1
        if restore_from:
            _copy_checkpoint(restore_from, model_path, restore_step)
            _restore_model(sess, model_path, restore_step)
            # evaluate the pretrained model
            step = restore_step
            _iterate_dataset(sess, model, valid_iter, valid_hd, valid_wtr, step)
            _iterate_dataset(sess, model, test_iter, test_hd, test_wtr, step)
            step += 1
        else:
            sess.run(tf.global_variables_initializer())
            step = 1

        if profiling:
            _profile_and_exit(sess, model, optim, train_hd)

        pbar = tqdm.tqdm(total=save_every, desc='Train', unit='batch')
        best_loss, steps_no_improve = 1e10, 0
        selected_optimizer = optimizer_names[0]
        patiences = optimization_params.get('switch_optimizer_patience_steps', [None])
        try:
            while True:
                if step % record_every == 0:
                    # decide whether to switch optimizer
                    check_patiences = [i+1 for i, p in enumerate(patiences) if steps_no_improve > p]
                    if check_patiences:
                        new_optimizer_index = check_patiences[0]
                        new_optimizer_name = optimizer_names[new_optimizer_index]
                        if selected_optimizer != new_optimizer_name:
                            log.warn('switching optimizer to: {}'.format(new_optimizer_name))
                            selected_optimizer = new_optimizer_name

                    summary, _, loss = sess.run(
                            [train_summary, optim, model.loss],
                            feed_dict={model.handle: train_hd,
                                       model.keep_prob: keep_prob,
                                       'optimizer_name:0': selected_optimizer})
                    pbar.set_postfix(loss='{:.3f}'.format(loss))
                    train_wtr.add_summary(summary, step)

                    # NOTE: we may need to apply this on validation data
                    if loss < best_loss:
                        steps_no_improve = 0
                        best_loss = loss
                    else:
                        steps_no_improve += record_every
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


def test(name: str,
         model_type: str,
         step: int = None,
         mode: str = 'test',
         data_seed: int = None,
         data_name: str = 'SNLI',
         data_embedding: str = 'GloVe',
         data_pad: bool = True,
         batch_size: int = 10,
         optimization_params: dict = {},
         **kwargs,
         ) -> None:
    model_path = build.get_model_path(name)

    model = getattr(nn, model_type)(
            embeddings=data.load_embeddings(data_name, data_embedding, data_seed),
            **kwargs)
    log.info(str(model))
    log.debug('Model parameters:\n\n\t' +
              '\n\t'.join(graph.print_trainable_variables().split('\n')))

    with tf.Session(config=_make_config()) as sess:
        data_iter, data_hd = _make_dataset_iterator(
                type_name='initializable_iterator',
                handle_name='data_handle',
                dataset=data.load_dataset(
                        data_name, mode, data_embedding, data_seed),
                batch_size=batch_size,
                shuffle=False,
                pad=data_pad,
                session=sess)

        _restore_model(sess, model_path, step)

        y_preds, y_trues = [], []  # type: ignore
        sess.run(data_iter.initializer)
        while True:
            try:
                true, pred = sess.run(
                        [model.y, model.prediction],
                        feed_dict={model.handle: data_hd,
                                   model.keep_prob: 1.0})
                y_preds.extend(np.squeeze(pred).tolist())
                y_trues.extend(np.squeeze(true).tolist())
            except tf.errors.OutOfRangeError:
                break

    # accuracy
    print('Acc: %.4f' % sklearn.metrics.accuracy_score(y_trues, y_preds))


if __name__ == "__main__":
    # Disable the debugging INFO and WARNING information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Testing reads arguments from both file and commend line. Commend line
    # arguments can override ones parsed from file.
    mode = sys.argv[1]
    kwargs = parse.parse_args(sys.argv[1:])
    if 'file' in kwargs:
        # Use the file name as the default model name.
        fname = os.path.basename(kwargs['file'])  # type: ignore
        if 'name' not in kwargs:
            kwargs['name'] = fname[:fname.rfind('.')]  # type: ignore
        kwargs = {**parse.parse_yaml(kwargs['file'], mode=mode), **kwargs}
        del kwargs['file']

    log.debug('Input arguments:\n\n\t%s\n' %
              '\n\t'.join('%-20s %s' % (k + ':', v) for k, v in kwargs.items()))

    locals()[mode](**kwargs) # type: ignore
