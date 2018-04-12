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
         data_name: str = 'MSRP',
         data_preproc: str = 'Tokenize',
         embedding: str = 'Word2Vec',
         load: bool = True,
         visualize: bool = False,
         softmax: bool = False,
         **kwargs,
         ) -> None:
    print()
    print(data_name, mode)
    print(kwargs)
    model_path = build.get_model_path(name)
    test_data = data.load(data_name, mode, data_preproc, embedding, 1)
    test_data.reset_max_len(41)  # TODO

    model = nn.Decomposeable(word_embeddings=test_data.embeds,
                             seq_len=test_data.max_len,
                             **kwargs)
    _print_model_setup(model)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        if load:
            print("reading from: %s" % (model_path))
            tf.train.Saver().restore(sess, build.get_saved_model(model_path))
            print("%s restored." % (model_path))

        #print(sess.run(model.evaluator.factor))

        preds, x1_atts, x2_atts, x1_sals, x2_sals = evaluate(
                sess, model, test_data, mode.title())

    test_cases = [] # type: list
    for i in range(test_data.data_size):
        test_cases += [test_data.labels[i], preds[i],
                       test_data.s1s[i], test_data.s2s[i],
                       test_data.w1s[i], test_data.w2s[i],
                       x1_atts[i], x2_atts[i],
                       x1_sals[i], x2_sals[i]],

    if visualize:
        tsm = vis.TextSaliencyMap()
        for gt, pred, w1_all, w2_all, w1, w2, x1_att, x2_att, x1_sal, x2_sal in test_cases:
            tsm.add_colored_text([to_attented_text(w1_all, w1, x1_att, softmax),
                                  to_attented_text(w2_all, w2, x2_att, softmax)],
                                 pred, gt)
            tsm.add_colored_text([to_attented_text(w1_all, w1, x1_sal, softmax),
                                  to_attented_text(w2_all, w2, x2_sal, softmax)],
                                 pred, gt)
        tsm.write(os.path.join(model_path, 'visualize-%s.html' % mode))

    # accuracy
    y_trues, y_preds = [], []  # type: list, list
    for gt, pred, _, _, _, _, _, _, _, _ in test_cases:
        y_trues += gt, 
        y_preds += pred,
    print('Acc: %.4f' % sklearn.metrics.accuracy_score(y_trues, y_preds))
    print('F1:  %.4f' % sklearn.metrics.f1_score(y_trues, y_preds))


if __name__ == "__main__":
    # Disable the debugging INFO and WARNING information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Testing reads arguments from both file and commend line. Commend line
    # arguments can override ones parsed from file.
    kwargs = parse.parse_args(sys.argv)
    if 'file' in kwargs:
        kwargs = {**parse.parse_yaml(kwargs['file'], mode='test'), **kwargs}
    test(**kwargs)
