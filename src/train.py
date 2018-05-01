""" Model training. """

import os
import sys

import tensorflow as tf
import tqdm
import shutil

import embed
import data
import nn
from util import build, parse, graph, log
from util.annotation import print_section


@print_section
def _print_trainable_variables():
    graph.print_trainable_variables()
    #print("List of Variables:")
    #for v in tf.trainable_variables():
    #    print(v.name)


@print_section
def _print_number_of_variables(model):
    print("Total Variables: %d" % model.count_parameters())


@print_section
def _print_model_setup(model):
    print(model)


def train(name: str,
          batch_size: int = 256,
          epoch_num: int = 200,
          learning_rate: float = 0.05,
          data_name: str = 'SNLI',
          data_embedding: str = 'GloVe',
          validate: bool = True,
          **kwargs
          ) -> None:
    print(kwargs)

    # Data preparation
    model_path = build.get_model_path(name)
    print('save path:', build.get_save_path(model_path))
    shutil.rmtree(model_path, ignore_errors=True)  # remove previous trained
    train_data = data.load(data_name, 'train', data_embedding)
    valid_data = data.load(data_name, 'validation', data_embedding)

    # Network setup
    model = nn.Decomposeable(dataset=train_data, **kwargs)
    _print_model_setup(model)
    _print_trainable_variables()
    _print_number_of_variables(model)

    # Summary setup
    tf.summary.scalar("Loss", model.loss)
    tf.summary.scalar("Accuracy", model.performance)
    valid_summary = tf.summary.merge_all()

    # Run meta setup
    # NOTE only available with CUDA == 8.0
    #run_metadata = tf.RunMetadata()

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
        train_wtr = tf.summary.FileWriter(os.path.join(model_path, 'train'),
                                          sess.graph)
        if validate:
            valid_wtr = tf.summary.FileWriter(os.path.join(model_path, 'valid'))
        sess.run(init)
        step = 0
        pbar = tqdm.tqdm(range(1, epoch_num + 1), desc='Train', unit='epoch')
        for e in pbar:
            #options=tf.RunOptions(
            #        trace_level=tf.RunOptions.FULL_TRACE),
            #run_metadata=run_metadata

            # Training
            sess.run(model.init)
            while True:
                try:
                    if not step % 100:
                        summary, _, loss = sess.run(
                                [train_summary, optimizer, model.loss])
                        pbar.set_postfix(loss='{:.3f}'.format(loss))
                        train_wtr.add_summary(summary, step)
                        #train_wtr.add_run_metadata(run_metadata, 'step%05d' % step)
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
        fname = os.path.basename(kwargs['file'])
        kwargs['name'] = fname[:fname.rfind('.')]
        kwargs = {**parse.parse_yaml(kwargs['file'], mode='train'), **kwargs}
        del kwargs['file']
    train(**kwargs)
