from abc import ABC, abstractmethod
import math
import os
import pickle
import random
import re
import typing as t

import numpy as np
import nltk
import tensorflow as tf

import embed
from util import build
from util.log import exec_log as log


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


class MSRP(Dataset):
    DATA_FILES = {'train':      ['data/MSRP/MSRP-train.tsv',
                                 'data/MSRP/MSRP-dev.tsv'],
                  'validation': ['data/MSRP/MSRP-dev.tsv'],
                  'test':       ['data/MSRP/MSRP-test.tsv'],
                  }

    @classmethod
    def parse(cls, mode: str) \
            -> t.Generator[t.Tuple[str, str, t.Union[int, float], t.Any, t.Any],
                           None,
                           None]:
        for data_file in cls.DATA_FILES[mode]:
            with open(data_file, 'r', encoding='utf-8') as f:
                f.readline()  # skip the heading line
                for line in f:
                    label, _, _, s1, s2 = line[:-1].split('\t')
                    yield s1, s2, int(label), _, _


class ParseTree:
    def __init__(self, tag):
        self.tag = tag
        self.word = None
        self.children = []

    def __repr__(self):
        children = ' '.join(map(repr, self.children)) if self.children else ''
        return '(%s%s%s)' % (self.tag,
                             ' ' + self.word if self.word else '',
                             ' ' + children if children else '')

    def get_leafs(self):
        """ Represents the leaf nodes as a list of words and a list of
        crosspoonding tags. """
        words, tags = [], []
        def postorder(tree):
            if tree:
                for c in tree.children:
                    postorder(c)
                if tree.word:
                    words.append(tree.word)
                    tags.append(tree.tag)
        postorder(self)
        return words, tags

    def get_internals(self, drop_size=None):
        """ Represents the internal nodes as a list of direct children tuples
        and crossponding tags. For example, given a parse tree

                 6 (ROOT)
                 /      \
              4 (NP)   5 (VP)
               /       /   \
            1 (NN)  2 (V)  3 (NN)

        This function returns: [[1], [2, 3], [4, 5]], [NP, VP, ROOT].

        Args:
            drop_size: drop the internal nodes and their parents with size
                greater than the drop size.
        """
        nodes, tags = [], []
        self.__top_leaf = 0
        self.__top_intl = len(self.get_leafs()[0])
        def postorder(tree):
            if tree:
                cidx = [postorder(c) for c in tree.children]
                if not all(cidx):
                    return 0
                if tree.word:
                    self.__top_leaf = self.__top_leaf + 1
                    return self.__top_leaf
                else:
                    if drop_size and len(tree.children) > drop_size:
                        return 0
                    nodes.append(tuple(cidx))
                    tags.append(tree.tag)
                    self.__top_intl = self.__top_intl + 1
                    return self.__top_intl
        postorder(self)
        return nodes, tags


    @staticmethod
    def parse(string) -> 'ParseTree':
        """ Parse the syntax tree given the sentence parse marker. This
        function returns two lists of tokens and crossponding POS tags,
        respectively. """
        unit_ptn = r'(\(|\)|[^()\s]+)'
        units = re.findall(unit_ptn, string)
        idx = 0
        stk = []  # type: list
        word = None
        while True:
            curr = units[idx]
            if curr == '(':
                stk += ParseTree(units[idx + 1]),
                idx += 1
            elif curr == ')':
                node = stk.pop()
                if stk:
                    stk[-1].children += node,
                else:
                    return node
            else:
                stk[-1].word = curr
            idx += 1


class SNLI(Dataset):
    DATA_FILES = {'train':      ['data/SNLI/SNLI-train.tsv'],
                  'validation': ['data/SNLI/SNLI-dev.tsv'],
                  'test':       ['data/SNLI/SNLI-test.tsv'],
                  }

    LABELS = {'neutral':       0,
              'contradiction': 1,
              'entailment':    2}

    TEMP_DROP_VAL = 4

    def __init__(self, *args, **kwargs)-> None:
        super(SNLI, self).__init__(*args, **kwargs)
        self.tags = {}  # type: t.Dict[str, int]
        for mode in ['train', 'validation', 'test']:
            for _, _, label, (_, tags1), (_, tags2) in self.parse(mode):
                for tag in tags1 + tags2:
                    if tag not in self.tags:
                        # Starts tag ID at 1 to differenciate with the zero
                        # paddings
                        self.tags[tag] = len(self.tags) + 1
        for feats in self.x1_feats + self.x2_feats:
            # Pad template to 3-element-tuples
            for i in range(len(feats[0])):
                feats[0][i] = list(feats[0][i]) \
                            + [0] * (self.TEMP_DROP_VAL - len(feats[0][i]))
            if not feats[0]:
                # To aviod zero shape error in TF
                feats[0] = [[0] * self.TEMP_DROP_VAL]
            # Transform tags to tag IDs
            feats[1] = [self.tags[t] for t in feats[1]]

    @classmethod
    def parse(cls, mode: str) \
            -> t.Generator[t.Tuple[str, str, t.Union[int, float], t.Any, t.Any],
                           None,
                           None]:
        for data_file in cls.DATA_FILES[mode]:
            with open(data_file, 'r', encoding='utf-8') as f:
                f.readline()  # skip the heading line
                for line in f:
                    fields  = line.strip().split('\t')
                    label = cls.LABELS.get(fields[0], None)
                    if label is not None:  # skip the non-relation pairs
                        def process(field):
                            tree = ParseTree.parse(field)
                            words, w_tags = tree.get_leafs()
                            intls, i_tags = tree.get_internals(cls.TEMP_DROP_VAL)
                            return ' '.join(words), intls, w_tags + i_tags
                        s1, temp1, tags1 = process(fields[3])
                        s2, temp2, tags2 = process(fields[4])
                        yield (s1, s2, label, [temp1, tags1], [temp1, tags2])

    @classmethod
    def _tokenize(cls, sentence: str) -> t.List[str]:
        """ Split the tokens as the SNLI dataset has already been parsed. Pad a
        EOS symbol to the end of the sentence. """
        return sentence.split()# + ['<EOS>']

    @classmethod
    def _oov_assign(cls, word: str, dim: int) -> np.array:
        """ Assign one of 100 randomly generated vector to each OOV word.

        same embedding vector.
        """
        def hash_val(word):
            val = 1
            for c in word:
                val = val * 31 + ord(c)
            return val
        if word == '<EOS>':
            if cls._EOS_EMBED is None:
                cls._EOS_EMBED = cls._gen_random_embed(dim)
            return cls._EOS_EMBED
        if word not in cls._OOV_MAP:
            if not cls._OOV_EMBEDS:
                cls._OOV_EMBEDS = [cls._gen_random_embed(dim)
                                   for _ in range(100)]
            cls._OOV_MAP[word] = cls._OOV_EMBEDS[hash_val(word) % 100]
        return cls._OOV_MAP[word]
    _EOS_EMBED = None
    _OOV_EMBEDS = []  # type: t.List[np.array]
    _OOV_MAP = {}  # type: t.Dict[str, np.array]

    @classmethod
    def _gen_random_embed(cls, dim):
        """ Generate an embedding vector of Gausian distributed value with 0
        mean and 0.1 standard deviation. The return value takes its l2-norm
        form. """
        embed = (np.random.randn(1, dim) * 0.1).astype("float32")
        return embed / np.linalg.norm(embed)


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
