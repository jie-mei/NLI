from abc import ABC, abstractmethod
import math
import os
import pickle
import re
from typing import List, Set, Tuple, Generator, Union

import numpy as np
import nltk
import tensorflow as tf

import embed
from util import build
import preproc


class Dataset(ABC):
    """
    Attributes:
        embeds:
        s1s:
        s2s:
        x1s:
        x2s:
        labels:
        features:
        batch_size:
        embed:
    """
    def __init__(self,
                 mode: str,
                 data_preproc: preproc.DataPreproc,
                 word_embedding: embed.WordEmbedding,
                 batch_size: int) -> None:
        self._index = 0
        self.batch_size = batch_size
        # Load data from file
        vocab = set()  # type: Set[str]
        self.s1s, self.s2s = [], [] # type: List[List[str]], List[List[str]]
        self.w1s, self.w2s = [], [] # type: List[List[str]], List[List[str]]
        self.labels = [] # type: List[Union[int, float]]
        # Load indexed word embedding
        iwe = self.__load_indexed_word_embedding(data_preproc, word_embedding)
        self.embeds = iwe.embeds
        # Preprocess the dataset
        def preproc(text, orig_list, proc_list):
            orig_tks = nltk.word_tokenize(text)
            proc_tks = data_preproc.preproc(orig_tks)
            orig_list.append(orig_tks)
            proc_list.append(proc_tks)
        for s1, s2, label in self.parse(mode):
            preproc(s1, self.s1s, self.w1s)
            preproc(s2, self.s2s, self.w2s)
            self.labels.append(label)
        # Remove empty strings and convert words to ids
        def word2id(word_lists: List[List[str]]) -> List[List[int]]:
            return [[iwe.get_id(w) for w in words if w] for words in word_lists]
        self.x1s, self.x2s = word2id(self.w1s), word2id(self.w2s)
        # Summarize
        self.max_len = max(len(s) for s in self.x1s + self.x2s)
        self.data_size = len(self.x1s)
        # Align input word ids with tailing zeros
        def align_ndarray(xs: List[List[int]]):
            lists = [np.expand_dims(
                            np.pad(l, [0, self.max_len - len(l)], 'constant'),
                            axis=0)
                     for l in xs]
            return np.concatenate(lists)
        self.x1s, self.x2s = align_ndarray(self.x1s), align_ndarray(self.x2s)
        # Convert lists to numpy arrays
        self.s1s, self.s2s = np.array(self.s1s), np.array(self.s2s)
        self.w1s, self.w2s = np.array(self.w1s), np.array(self.w2s)
        self.labels = np.array(self.labels)

    def create_tf_dataset(self, shuffle_buffer_size=10240, shuffle=True, repeat_num=1, truncated_seq=40):
        if truncated_seq is not None:
            self.x1s = self.x1s[:, :truncated_seq]
            self.x2s = self.x2s[:, :truncated_seq]
        np_data_slices = (self.x1s, self.x2s, self.labels)
        dataset = tf.data.Dataset.from_tensor_slices(np_data_slices)
        dataset = dataset.batch(self.batch_size)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10240)
        self.dataset = dataset.repeat(repeat_num)

        self.iterator = tf.data.Iterator.from_structure(
            self.dataset.output_types, self.dataset.output_shapes)
        tensors = self.iterator.get_next()
        x1, x2, y = tensors
        self.x1 = tf.cast(x1, tf.int32)
        self.x2 = tf.cast(x2, tf.int32)
        self.y = tf.cast(y, tf.int32)

        self.initializer = self.iterator.make_initializer(self.dataset)

    @classmethod
    def __load_indexed_word_embedding(cls,
            data_preproc: preproc.DataPreproc,
            word_embedding: embed.WordEmbedding) -> embed.IndexedWordEmbedding:
        # Load preprocessed word embeddings from pkl if applicable.
        pkl_path = os.path.join(build.BUILD_DIR, 'data',
                '{}.{}.{}.pkl'.format(cls.__name__,
                                      data_preproc.__class__.__name__,
                                      word_embedding.__class__.__name__))
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as pkl_file:
                embeds = pickle.load(pkl_file)
        else:
            vocab = set()  # type: Set[str]
            for mode in ['train', 'validation', 'test']:
                for s1, s2, label in cls.parse(mode):
                    vocab.update(data_preproc.preproc(nltk.word_tokenize(s1)))
                    vocab.update(data_preproc.preproc(nltk.word_tokenize(s2)))
            embeds = embed.IndexedWordEmbedding(vocab, word_embedding)
            # Serialize for reusing
            os.makedirs(os.path.normpath(os.path.join(pkl_path, os.pardir)),
                    exist_ok=True)
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(embeds, pkl_file, 4)
        return embeds

    def reset_max_len(self, max_len):
        if max_len > self.max_len:
            pad = max_len - self.max_len
            self.x1s = np.pad(self.x1s, [[0, 0], [0, pad]], 'constant')
            self.x2s = np.pad(self.x2s, [[0, 0], [0, pad]], 'constant')
        elif max_len < self.max_len:
            self.x1s = self.x1s[:, : max_len]
            self.x2s = self.x2s[:, : max_len]
        self.max_len = max_len

    @classmethod
    @abstractmethod
    def parse(cls, mode: str) \
            -> Generator[Tuple[str, str, Union[int, float]], None, None]:
        """ Generate parsed data. """
        pass

    def reset_index(self) -> None:
        # Shuffle data instances
        p = np.random.permutation(len(self.x1s))
        self.s1s, self.s2s = self.s1s[p], self.s2s[p]
        self.w1s, self.w2s = self.w1s[p], self.w2s[p]
        self.x1s, self.x2s = self.x1s[p], self.x2s[p]
        self.labels = self.labels[p]
        # Reset index
        self._index = 0

    def num_batches(self) -> int:
        return math.ceil(len(self.labels) / self.batch_size)

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray, list]:
        num = min(self.data_size - self._index, self.batch_size)
        batch_x1s = self.x1s[self._index: self._index + num]
        batch_x2s = self.x2s[self._index: self._index + num]
        batch_labels = self.labels[self._index: self._index + num]
        self._index += num
        return batch_x1s, batch_x2s, batch_labels


class MSRP(Dataset):
    DATA_FILES = {'train':      ['data/MSRP/MSRP-train.tsv',
                                 'data/MSRP/MSRP-dev.tsv'],
                  'validation': ['data/MSRP/MSRP-dev.tsv'],
                  'test':       ['data/MSRP/MSRP-test.tsv'],
                  }

    @classmethod
    def parse(cls, mode: str) \
            -> Generator[Tuple[str, str, Union[int, float]], None, None]:
        for data_file in cls.DATA_FILES[mode]:
            with open(data_file, 'r', encoding='utf-8') as f:
                f.readline()  # skip the heading line
                for line in f:
                    label, _, _, s1, s2 = line[:-1].split('\t')
                    yield s1, s2, int(label)


class WikiQA(Dataset):
    DATA_FILES = {'train':      ['data/WikiQA/WikiQA-train.tsv',
                                 'data/WikiQA/WikiQA-dev.tsv'],
                  'validation': ['data/WikiQA/WikiQA-dev.tsv'],
                  'test':       ['data/WikiQA/WikiQA-test.tsv'],
                  }

    @classmethod
    def parse(cls, mode: str) \
            -> Generator[Tuple[str, str, Union[int, float]], None, None]:
        for data_file in cls.DATA_FILES[mode]:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    s1, s2, label = line.strip().split('\t')
                    yield s1, s2, int(label)


class SNLI(Dataset):
    DATA_FILES = {'train':      ['data/SNLI/SNLI-train.tsv'],
                  'validation': ['data/SNLI/SNLI-dev.tsv'],
                  'test':       ['data/SNLI/SNLI-test.tsv'],
                  }

    LABELS = {'neutral':       0,
              'contradiction': 1,
              'entailment':    2}

    @classmethod
    def parse(cls, mode: str) \
            -> Generator[Tuple[str, str, Union[int, float]], None, None]:
        def parse_sentence(sent):
            # Remove all brackets.
            return re.sub(r'(\(|\)) ?', '', sent)
        for data_file in cls.DATA_FILES[mode]:
            with open(data_file, 'r', encoding='utf-8') as f:
                f.readline()  # skip the heading line
                for line in f:
                    fields  = line.strip().split('\t')
                    label = cls.LABELS.get(fields[0], None)
                    if label is not None:  # skip the non-relation pairs
                        yield (parse_sentence(fields[1]),
                               parse_sentence(fields[2]),
                               label)


def load(data_name: str,
         data_mode: str,
         data_preproc_name: str = 'Tokenize',
         embedding_name: str = 'Word2Vec',
         batch_size: int = 0,
         ) -> Dataset:
    # Load preprocessed data object from pkl if applicable.
    pkl_path = os.path.join(build.BUILD_DIR, 'data',
            '{}-{}.{}.{}.pkl'.format(data_name,
                                     data_mode,
                                     data_preproc_name,
                                     embedding_name))
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as pkl_file:
            dataset = pickle.load(pkl_file)
    else:
        embedding = embed.get(embedding_name)
        data_preproc = getattr(preproc, data_preproc_name)()
        dataset = globals()[data_name](data_mode, data_preproc, embedding,
                batch_size)
        os.makedirs(os.path.normpath(os.path.join(pkl_path, os.pardir)),
                    exist_ok=True)
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(dataset, pkl_file, 4)
    dataset.batch_size = batch_size if batch_size else dataset.data_size
    return dataset
