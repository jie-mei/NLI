""" Model training. """

import os
import sys
import time

import tensorflow as tf
import tqdm
import shutil

import embed
import data
import nn
from util import build, parse, graph
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
          learning_rate: float = 0.02,
          data_name: str = 'MSRP',
          data_preproc: str = 'Tokenize',
          data_embedding: str = 'Word2Vec',
          validate: bool = True,
          **kwargs
          ) -> None:
    print(kwargs)
    model_name = kwargs.get('model_name', 'model')

    # Data preparation
    model_path = build.get_model_path(name, model_name)
    print('save path:', build.get_save_path(model_path))
    shutil.rmtree(model_path, ignore_errors=True)  # remove previous trained
    train_data = data.load(data_name, 'train', data_preproc, data_embedding, batch_size)
    train_data.create_tf_dataset(shuffle_buffer_size=20480)
    valid_data = data.load(data_name, 'validation', data_preproc, data_embedding)
    valid_data.create_tf_dataset(shuffle=False)
    valid_data.reset_max_len(train_data.max_len)

    # Network setup
    model = nn.Decomposeable(word_embeddings=train_data.embeds,
                             dataset=train_data,
                             train_mode=True,
                             **kwargs)
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

    # Optimization
    #optimizer = (tf.train.AdagradOptimizer(learning_rate, name="optimizer")
    #        .minimize(model.loss))
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.995, staircase=True)
    optimizer = get_train_op(model.loss, lr=learning_rate, clip_gradients=None, optimizer_name='adagrad', global_step=global_step)

    train_summary = tf.summary.merge_all()

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
        for e in tqdm.tqdm(range(1, epoch_num + 1)):
            # Training
            desc = 'Epoch {num:{w}}'.format(num=e, w=len(str(epoch_num)))
            sess.run(train_data.initializer)
            while True:
                try:
                    _ = sess.run(
                        [optimizer],
                        )
                    if step % 100 == 0:
                        summary, _, loss = sess.run(
                            [train_summary, optimizer, model.loss],
                            )
                        train_wtr.add_summary(summary, step)
                except tf.errors.OutOfRangeError:
                    break

                #train_wtr.add_run_metadata(run_metadata, 'step%05d' % step)
                step += 1

#            if validate:
#                valid_data.reset_index()
#                x1, x2, y = valid_data.next_batch()
#                summary, = sess.run([valid_summary],
#                        feed_dict={model.x1: x1, model.x2: x2, model.y: y})
#                valid_wtr.add_summary(summary, step)
#                valid_wtr.flush()
            save_path = (tf.train.Saver(max_to_keep=2)
                .save(sess, build.get_save_path(model_path), global_step=epoch_num))
        print("model saved as", save_path)

def get_train_op(loss, lr=0.01, clip_gradients=None, optimizer_name='adam', global_step=None):
    if optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif optimizer_name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(lr)
    else:
        raise ValueError('invalid optimizer name')

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    if clip_gradients is not None:
        grads, _ = tf.clip_by_global_norm(grads, clip_gradients)

    for g in grads:
        tf.summary.histogram(g.name, g)

    if global_step is None:
        global_step = tf.train.get_global_step()

    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=global_step)
    return train_op


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
    train(**kwargs)
