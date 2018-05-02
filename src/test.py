""" Test the model.
"""

import collections
import os
import sys
import numpy as np
import tensorflow as tf
import tqdm
from typing import List, Tuple, Union

import sklearn

import data
import data
import embed
import nn
from util import build, parse
from util import visualize as vis
from util.annotation import print_section
from util.log import exec_log as log



@print_section
def _print_model_setup(model):
    print(model)


@print_section
def evaluate(session: tf.Session,
             model: nn.Model,
             dataset: data.Dataset,
             title: str = 'Evaluation') \
         -> Tuple[List[Union[int, float]],
                  List[List[float]],
                  List[List[float]],
                  List[List[float]],
                  List[List[float]]]:
    preds = []  # type: List[Union[int, float]]
    for i in tqdm.trange(dataset.data_size, unit='instance', desc=title):
        x1, x2, y = dataset.next_batch()
        pred = session.run(
                [model.prediction],
                feed_dict={model.x1: x1, model.x2: x2, model.y: y})
        preds += np.squeeze(pred).tolist(),
    x1_atts, x2_atts, x1_sals, x2_sals = [[1] * len(preds) for _ in range(4)]
    return preds, x1_atts, x2_atts, x1_sals, x2_sals


@print_section
def _print_number_of_variables(model):
    print("Total Variables: %d" % model.count_parameters())


@print_section
def evaluate_vis(session: tf.Session,
             model: nn.Model,
             dataset: data.Dataset,
             title: str = 'Evaluation') \
         -> Tuple[List[Union[int, float]],
                  List[List[float]],
                  List[List[float]],
                  List[List[float]],
                  List[List[float]]]:
    preds = []  # type: List[Union[int, float]]
    x1_atts, x2_atts = [], []  # type: List[List[float]], List[List[float]]
    x1_sals, x2_sals = [], []  # type: List[List[float]], List[List[float]]
    #dataset.reset_index()
    for i in tqdm.trange(dataset.data_size, unit='instance', desc=title):
        x1, x2, y = dataset.next_batch()
        pred, x1_att, x2_att, x1_sal, x2_sal = session.run(
                [model.prediction, model.x1_att, model.x2_att,
                    model.x1_sal, model.x2_sal],
                feed_dict={model.x1: x1, model.x2: x2, model.y: y})
        preds += np.squeeze(pred).tolist(),
        x1_atts += np.squeeze(x1_att).tolist(),
        x2_atts += np.squeeze(x2_att).tolist(),
        x1_sals += np.squeeze(x1_sal).tolist(),
        x2_sals += np.squeeze(x2_sal).tolist(),
    return preds, x1_atts, x2_atts, x1_sals, x2_sals


def to_attented_text(all_words: List[str],
                     att_words: List[str],
                     att_vals: List[int],
                     softmax: bool
                     ) -> vis.AttentedText:
    #att_vals = np.mean(att_vals, axis=(0, 1)).tolist()
    #att_vals = att_vals.tolist()[0]
    all_vals = []
    vidx = 0
    for w in att_words:
        if not w:
            all_vals.append(0)
        else:
            all_vals.append(att_vals[vidx])
            vidx += 1
    if softmax:
        exp_all_vals = np.exp(all_vals - np.max(all_vals))
        all_vals = exp_all_vals / exp_all_vals.sum(axis=0)
        all_vals = all_vals.tolist()
    return vis.AttentedText(all_words, all_vals)


def test(name: str,
         mode: str = 'test',
         data_name: str = 'SNLI',
         data_embedding: str = 'GloVe',
         **kwargs,
         ) -> None:
    print()
    print(data_name, mode)
    print(kwargs)
    model_path = build.get_model_path(name)
    test_data = data.load_dataset(data_name, mode, data_embedding)

    model = nn.Decomposeable(dataset=test_data, **kwargs)
    _print_model_setup(model)
    _print_number_of_variables(model)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        log.info('Restore pre-trained model from: %s' % model_path)
        tf.train.Saver().restore(sess, build.get_saved_model(model_path))

        y_preds, y_trues = [], []
        sess.run(model.init)
        while True:
            try:
                true, pred = sess.run([model.y, model.prediction])
                y_preds.extend(np.squeeze(pred).tolist())
                y_trues.extend(np.squeeze(true).tolist())
            except tf.errors.OutOfRangeError:
                break

    # accuracy
    print('Acc: %.4f' % sklearn.metrics.accuracy_score(y_trues, y_preds))
    #print('F1:  %.4f' % sklearn.metrics.f1_score(y_trues, y_preds))


if __name__ == "__main__":
    # Disable the debugging INFO and WARNING information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Testing reads arguments from both file and commend line. Commend line
    # arguments can override ones parsed from file.
    kwargs = parse.parse_args(sys.argv)
    if 'file' in kwargs:
        # Use the file name as the default model name.
        fname = os.path.basename(kwargs['file'])
        kwargs['name'] = fname[:fname.rfind('.')]
        kwargs = {**parse.parse_yaml(kwargs['file'], mode='test'), **kwargs}
        del kwargs['file']
    test(**kwargs)
