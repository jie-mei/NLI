from abc import ABC, abstractmethod
import os
import pickle
import random
import typing as t

import numpy as np
import nltk

import embed
from util import build
from util.log import exec_log as log


# TODO:
#  * Avoid the Dataset being parsed multiple times in preprocessing.


class Dataset(ABC):
    """
    Attributes:
        x1_words:
        x2_words:
        x1_ids:
        x2_ids:
        x1_feats:
        x2_feats:
        labels:
        word_embedding:
    """
    def __init__(self,
            mode: str,
            word_embedding: embed.WordEmbedding = None,
            indexed_word_embedding: embed.IndexedWordEmbedding = None,
            seed: int = None
            ) -> None:
        # Load indexed word embedding
        if indexed_word_embedding:
            self.word_embedding = indexed_word_embedding
        elif word_embedding:
            self.word_embedding = self._init_indexed_word_embedding(
                    word_embedding, seed)
        else:
            raise ValueError('Either word_embedding or indexed_word_embedding'
                             'should be provided.')
        # Load data from file
        vocab = set()  # type: t.Set[str]
        self.x1_words, self.x2_words = [], []  # type: t.List[t.List[str]], t.List[t.List[str]]
        self.x1_ids, self.x2_ids = [], []  # type: t.List[t.List[int]], t.List[t.List[int]]
        self.x1_feats, self.x2_feats = [], []  # type: t.List[t.List[t.Any]], t.List[t.List[t.Any]]
        self.labels = []  # type: t.List[t.Union[int, float]]
        def preproc(x_text_in, x_feats_in, x_words_out, x_ids_out, x_feats_out):
            words = self._tokenize(x_text_in)
            x_words_out.append(words)
            x_ids_out.append([self.word_embedding.get_id(w) for w in words])
            x_feats_out.append(x_feats_in)
        for text1, text2, label, feats1, feats2 in self.parse(mode):
            preproc(text1, feats1, self.x1_words, self.x1_ids, self.x1_feats)
            preproc(text2, feats2, self.x2_words, self.x2_ids, self.x2_feats)
            self.labels.append(label)

    @property
    def seed(self):
        return self.word_embedding.seed

    @classmethod
    def _init_indexed_word_embedding(cls,
            word_embedding: embed.WordEmbedding,
            seed: int
            )-> embed.IndexedWordEmbedding:
        vocab = set()  # type: t.Set[str]
        for mode in ['train', 'validation', 'test']:
            for s1, s2, label, _, _ in cls.parse(mode):
                vocab.update(cls._tokenize(s1))
                vocab.update(cls._tokenize(s2))
        return embed.IndexedWordEmbedding(vocab, word_embedding,
                lambda w: cls._oov_assign(w, word_embedding.dim), seed)

    @classmethod
    def _tokenize(cls, sentence: str) -> t.List[str]:
        """ A tokenization function for parsing the input text.

        The returning word list will be feed into the model. This method will be
        called by other methods. Thus, preprocessing steps, e.g. filtering or
        stemming, can be optionally applied by overriding this method.
        """
        return nltk.word_tokenize(sentence)

    @classmethod
    def _oov_assign(cls, word: str, dim: int) -> np.array:
        """ Return a embedding vector for an OOV word.

        Different assignment functions can be applied by overriding this method.
        The default function returns a fixed vector which entries are uniformly 
        distributed random values in [-0.1, 0.1].
        """
        if not cls._OOV:
            cls._OOV = np.random.uniform(-.1, .1, dim).astype("float32")
        return cls._OOV
    _OOV = None

    @classmethod
    @abstractmethod
    def parse(cls, mode: str) \
            -> t.Generator[t.Tuple[str, str, t.Union[int, float], t.Any, t.Any],
                           None,
                           None]:
        """ Parse texts and label of each record from the data file. """
        pass


def load_dataset(
        data_name: str,
        data_mode: str,
        embedding_name: str,
        seed: int = None
        ) -> Dataset:
    if seed:
        pkl_path = os.path.join(build.BUILD_DIR, 'data',
                '{}-{}.{}.seed{}.pkl'.format(
                        data_name, data_mode, embedding_name, seed))
    else:
        pkl_path = os.path.join(build.BUILD_DIR, 'data',
                '{}-{}.{}.pkl'.format(data_name, data_mode, embedding_name))
    # Load preprocessed data object from pkl if applicable.
    if os.path.exists(pkl_path):
        log.info('Restore %s %s dataset from file: %s' %
                 (data_name, data_mode, pkl_path))
        with open(pkl_path, 'rb') as pkl_file:
            dataset = pickle.load(pkl_file)
    else:
        log.info('Build %s %s dataset' % (data_name, data_mode))
        embedding = load_embeddings(data_name, embedding_name, seed)
        dataset = globals()[data_name](data_mode,
                indexed_word_embedding=embedding,
                seed=seed)
        os.makedirs(os.path.normpath(os.path.join(pkl_path, os.pardir)),
                    exist_ok=True)
        log.info('Serialize %s %s dataset to file %s.' %
                 (data_mode, data_name, pkl_path))
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(dataset, pkl_file, 4)
    return dataset


def load_embeddings(data_name: str, embedding_name: str, seed: int = None):
    """ Load a indexed word embedding object.
    
    This method will first check if the given setup have previously been
    serialized, restore the object from file. Otherwise, construct a new object.
    """
    # The naming convention of the serialization path.
    if seed:
        pkl_path = os.path.join(build.BUILD_DIR, 'data',
                '{}.{}.seed{}.pkl'.format(data_name, embedding_name, seed))
    else:
        pkl_path = os.path.join(build.BUILD_DIR, 'data',
                '{}.{}.pkl'.format(data_name, embedding_name))
    if os.path.exists(pkl_path):
        log.info('Restore corpus-specific indexed word embedding from file: %s'
                 % pkl_path)
        with open(pkl_path, 'rb') as pkl_file:
            embeds = pickle.load(pkl_file)
    else:
        log.info('Build corpus-specific indexed word embedding.')
        vocab = set()  # type: t.Set[str]
        embeds = globals()[data_name]._init_indexed_word_embedding(
                embed.init(embedding_name), seed)
        # Serialize for reusing
        log.info('Save corpus-specific indexed word embedding to file: %s.'
                 % pkl_path)
        os.makedirs(os.path.normpath(os.path.join(pkl_path, os.pardir)),
                exist_ok=True)
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(embeds, pkl_file, 4)
    return embeds
