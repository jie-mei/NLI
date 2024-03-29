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

import data
import nn
import optim
from util import build, parse, graph
from util.log import exec_log as log


def _select_kwargs_regex(kwargs, regex, invert=False, start=0):
    ans = {}
    for k, v in kwargs.items():
        matches = re.match(regex, k)
        if (invert and not matches) or (not invert and matches):
            ans[k[start:]] = v
    return ans


def _make_dataset(
        dataset: data.Dataset,
        batch_size: int,
        argument: bool = False,
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
    output_shapes = ([None], [None], [],             # x1, x2, y
                     [], [],                         # len1, len2 
                     [None, nn.WORD_SEQ_LEN],        # char1
                     [None, nn.WORD_SEQ_LEN],        # char2
                     [None, 4], [None, 4],           # temp1, temp2
                     [None], [None])  # type: tuple  # tag1, tag2
    if argument:
        log.debug('Apply data argumentation')
    def gen():
        def to_ord_list(word):
            """ Convert the first 16 characters of the given word to their
            ordinal value. If the given word contains less than 16 words, pad
            the list to length 16 with 0. """
            out = list(map(ord, list(word)))
            while len(out) < nn.WORD_SEQ_LEN:
                out += 0,
            return out[:nn.WORD_SEQ_LEN]
        for x1, x2, y, w1, w2, (temp1, tag1), (temp2, tag2) in zip(
                dataset.x1_ids, dataset.x2_ids, dataset.labels,
                dataset.x1_words, dataset.x2_words,
                dataset.x1_feats, dataset.x2_feats):
            yield (x1, x2, y, len(x1), len(x2), list(map(to_ord_list, w1)),
                    list(map(to_ord_list, w2)), temp1, temp2, tag1, tag2)
            if argument:
                yield (x2, x1, y, len(x2), len(x1), list(map(to_ord_list, w2)),
                        list(map(to_ord_list, w1)), temp2, temp1, tag2, tag1)
    dset = tf.data.Dataset.from_generator(gen,
            output_types=(tf.int32,) * 11,
            output_shapes=output_shapes)
    if cache:
        log.debug('Cache dataset during computation')
        dset = dset.cache()
    else:
        log.debug('Do not cache dataset during computation')

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
            log.debug('Generate batches using '
                      'tf.contrib.data.bucket_by_sequence_length')
            dset = dset.apply(tf.contrib.data.bucket_by_sequence_length(
                    (lambda x1, x2, y, l1, l2, c1, c2, tmp1, tmp2, tag1, tag2:
                            tf.maximum(l1, l2)),
                    bucket_boundaries,
                    [batch_size] * (len(bucket_boundaries) + 1)))
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
    with tf.name_scope(handle_name):
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


def _make_optim_manager(optim_manager_type, loss_op, clip_norm,
        train_regex_list, kwargs):
    om = getattr(optim, optim_manager_type)(loss_op, clip_norm,
            var_list=_search_var_list(train_regex_list),  # None for all vars.
            **_select_kwargs_regex(kwargs,
                    regex=r'^optim_manager(?!_type)',
                    start=14))
    log.info(str(om))
    idx = 1
    if _select_kwargs_regex(kwargs, regex=r'^optim_(?!manager)'):
        # Construct optimizer with 'optim_*' arguments
        optim_kwargs = _select_kwargs_regex(kwargs,
                regex=r'^optim_(?!manager)',
                start=6)
        om.add_optimizer(**optim_kwargs)
    else:
        while True:
            # Construct optimizer with 'optim{idx}_*' arguments
            optim_kwargs = _select_kwargs_regex(kwargs,
                    regex=r'^optim%d_' % idx,
                    start=6 + len(str(idx)))
            if not optim_kwargs:
                break
            om.add_optimizer(**optim_kwargs)
            idx += 1
    return om


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
                               model.keep_prob: 1.0,
                               model.is_training: False})  # disable dropout
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
    return acc


def _profile_and_exit(session, model, optimizer, handle):
    """ Profile the run metadata at the first iteration and exit the program.
    """
    from tensorflow.python.client import timeline
    run_metadata = tf.RunMetadata()
    for i in range(5):
        session.run([optimizer],
                feed_dict={model.handle: handle,
                           model.keep_prob: 1.0,
                           model.is_training: True},
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
          optim_manager_type: str = 'NotChange',
          data_name: str = 'SNLI',
          data_embedding: str = 'GloVe',
          data_argument: bool = False,
          data_pad: bool = True,
          data_cache: bool = False,
          data_seed: int = None,
          record_every: int = 64000,
          validate_every: int = 640000,
          save_every: int = 6400000,
          restore_from: str = None,
          restore_step: int = None,
          profiling: bool = False,
          clip_norm: int = None,
          seed: int = None,
          debug: bool = False,
          **kwargs
          ) -> None:

    # Data preparation
    model_path = build.get_model_path(name)
    shutil.rmtree(model_path, ignore_errors=True)  # remove previous trained

    # Network setup
    model = getattr(nn, model_type)(
            embeddings=data.load_embeddings(data_name, data_embedding, data_seed),
            **_select_kwargs_regex(kwargs, r'^optim[0-9]*_', invert=True))
    log.info(str(model))
    log.debug('Model parameters:\n\n\t' +
              '\n\t'.join(graph.print_trainable_variables().split('\n')))

    # Control randomization
    if seed:
        log.info('Set random seed for data shuffling and graph computation: %d'
                % seed)
        tf.set_random_seed(seed)

    train_summary = _make_model_summary(model)

    with tf.Session(config=_make_config()) as sess:
        if debug:
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

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
                argument=data_argument,
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

        om = _make_optim_manager(optim_manager_type, model.loss, clip_norm,
                train_regex_list, kwargs)

        test_wtr = tf.summary.FileWriter(os.path.join(model_path, 'test'))
        train_wtr = tf.summary.FileWriter(os.path.join(model_path, 'train'),
                sess.graph)
        # Build a validation summary writer for each optimizer
        valid_wtr = {}
        for optim in om.optims:
            valid_wtr[optim.get_name()] = tf.summary.FileWriter(
                    os.path.join(model_path, 'valid-%s' % optim.get_name()))

        if restore_from:
            _copy_checkpoint(restore_from, model_path, restore_step)
            _restore_model(sess, model_path, restore_step)
            # Evaluate the pretrained model
            step = restore_step
            _iterate_dataset(sess, model, valid_iter, valid_hd,
                    valid_wtr[om.optim.get_name()], step)
            _iterate_dataset(sess, model, test_iter, test_hd, test_wtr, step)
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        if profiling:
            _profile_and_exit(sess, model, om.optim_op, train_hd)

        pbar = tqdm.tqdm(total=save_every, desc='Train', unit=' inst')
        try:
            while True:
                feed_dict={model.handle: train_hd,
                           model.keep_prob: keep_prob,
                           model.is_training: True}
                if om.feed_lr:
                    feed_dict[om.lr_op] = om.lr_val
                if step % record_every == 0:
                    summary, _, loss = sess.run(
                            [train_summary, om.optim_op, model.loss],
                            feed_dict=feed_dict)
                    pbar.set_postfix(loss='{:.3f}'.format(loss))
                    train_wtr.add_summary(summary, step)
                else:
                    sess.run([om.optim_op], feed_dict=feed_dict)

                if step and step % validate_every == 0:
                    pbar.set_description('Valid')
                    valid_acc = _iterate_dataset(
                            sess, model, valid_iter, valid_hd,
                            valid_wtr[om.optim.get_name()], step)
                    # Update upon the validation perfomance
                    om.update(valid_acc, step)
                    pbar.set_description('Test')
                    _iterate_dataset(
                            sess, model, test_iter, test_hd, test_wtr, step)
                    pbar.set_description('Train')

                if step and step % save_every == 0:
                    save_path = _save_model(sess, model_path, step)
                    pbar.set_description(save_path)
                    pbar.update(batch_size)
                    pbar.close()
                    pbar = tqdm.tqdm(total=save_every, desc='Train', unit=' inst')
                else:
                    pbar.update(batch_size)

                step += batch_size

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
         print_errors: bool = False,
         print_errors_limit: int = 10,
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
        dataset = data.load_dataset(data_name, mode, data_embedding, data_seed)

        data_iter, data_hd = _make_dataset_iterator(
                type_name='initializable_iterator',
                handle_name='data_handle',
                dataset=dataset,
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
                                   model.keep_prob: 1.0,
                                   model.is_training: False})
                y_preds.extend(np.squeeze(pred).tolist())
                y_trues.extend(np.squeeze(true).tolist())
            except tf.errors.OutOfRangeError:
                break

    # print accuracy
    print('Acc: %.4f' % sklearn.metrics.accuracy_score(y_trues, y_preds))

    # Print confusion matrix
    labels = list(sorted(data.SNLI.LABELS.keys(),
                         key=lambda x: data.SNLI.LABELS[x]))
    cm = sklearn.metrics.confusion_matrix(y_trues, y_preds,
                                          labels=range(len(labels)))
    tmpl = '%15s ' * (len(labels) + 2)
    print(tmpl % tuple([''] + labels + ['']))
    corr = 0
    for i in range(len(labels)):
        stats = cm[i]
        prob = stats[i] / sum(stats)
        corr += stats[i]
        print(tmpl % tuple([labels[i]] + list(map(str, cm[i])) + ['%.4f' % prob]))
    print(tmpl % tuple(['%d / %d' % (corr, len(y_trues))] +
                       [''] * len(labels) +
                       ['%.4f' % (corr / len(y_trues))]))

    # Print errors
    if print_errors:
        tmpl = '\n%4d. Pred: %-20s  True: %s\n      %s\n      %s'
        for i, (y_pred, y_true) in enumerate(zip(y_preds, y_trues)):
            if y_pred != y_true and print_errors_limit != 0:
                s1 = ' '.join(dataset.x1_words[i])
                s2 = ' '.join(dataset.x2_words[i])
                l_pred = labels[y_pred]
                l_true = labels[y_true]
                print(tmpl % (i, l_pred, l_true, s1, s2))
                print_errors_limit -= 1


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
              '\n\t'.join('%-25s %s' % (k + ':', v) for k, v in kwargs.items()))

    locals()[mode](**kwargs) # type: ignore
