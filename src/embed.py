from abc import ABC, abstractmethod
import os
from typing import Dict, Set, Union, Callable

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import spacy

from util.log import exec_log as log


class OOVError(Exception):
    """ This error indicates the querying word is out-of-vocabulary. """
    pass


class WordEmbedding(ABC):
    """ An abstract collection of embeded words.

    Attributes:
        dim: The number of dimensions for each embedding vector.
        vocab: A set of word strings.
    """
    def __init__(self, dim: int, vocab: Set[str] = set()) -> None:
        self.dim = dim
        self.vocab = vocab

    @abstractmethod
    def get(self, word: str) -> np.ndarray:
        """ Get the embedding of the given word. """
        pass


class IndexedWordEmbedding(WordEmbedding):
    """ A collection of embedding words with unique IDs for looking up.

    Attributes:
        ids (Dict[str, int]): A mapping from word to id.
        embeds (np.array): A two dimentional array, where `embed[i]` stores the
            embedding for the word with id `i`.
    """
    def __init__(self,
                 vocab: Set[str],
                 embedding: WordEmbedding,
                 oov_embed_fn: Callable[[str], np.ndarray]
                 ) -> None:
        super(IndexedWordEmbedding, self).__init__(embedding.dim, vocab)
        # Copy embeddings from the given word embedding object.
        self.__ids, self.__embeds = {}, None  # type: Dict[str, int], np.ndarray
        self.__is_oov = [False] * len(vocab)
        if vocab:
            self.__embeds = np.array([0] * embedding.dim)
            for wid, w in enumerate(vocab):
                try:
                    embed = embedding.get(w)
                except OOVError:
                    embed = oov_embed_fn(w)
                    self.__is_oov[wid] = True
                self.__ids[w] = wid
                self.__embeds = np.vstack([self.__embeds, embed])
            self.__embeds = self.__embeds[1:]

    def get_embeddings(self) -> np.ndarray:
        return self.__embeds

    def get_id(self, word: str) -> int:
        return self.__ids[word]

    def __get_word_id(self, word: Union[str, int]) -> int:
        return word if isinstance(word, int) else self.__ids[word]

    def is_oov(self, word: Union[str, int]) -> bool:
        return self.__is_oov[self.__get_word_id(word)]

    def get(self, word: Union[str, int]) -> np.array:
        return self.__embeds[self.__get_word_id(word)]


class PretrainedEmbedding(WordEmbedding):
    """ Pretrained word embeddings. """
    def __init__(self, path, dim, binary, lazy_initialization=False):
        super(PretrainedEmbedding, self).__init__(dim)
        self.path = path
        self.__binary = binary
        self.__model = None
        if not lazy_initialization:
            self.__load_model()

    def __load_model(self):
        log.info('Read pretrained %s embedding from file: %s' %
                 (self.__class__.__name__, self.path))
        self.__model = gensim.models.KeyedVectors.load_word2vec_format(
                self.path, binary=self.__binary)

    def get(self, word):
        if not self.__model:
            self.__load_model()
        if word in self.__model.vocab:
            return self.__model.word_vec(word)
        else:
            raise OOVError


class Word2Vec(PretrainedEmbedding):
    def __init__(self,
                 path='/home/jmei/data/GoogleNews-vectors-negative300.bin',
                 dim=300,
                 binary=True,
                 **kwargs):
        super(Word2Vec, self).__init__(path, dim, binary, **kwargs)


class GloVe(PretrainedEmbedding):
    def __init__(self,
                 path='/home/jmei/data/glove.840B.300d.txt',
                 dim=300,
                 binary=False,
                 **kwargs):
        # Preprocess the original GloVe data to allow using the gensim API.
        gensim_path = '{}.gensim.txt'.format(path[:-4])
        if not os.path.exists(gensim_path):
            glove2word2vec(path, gensim_path)
        super(GloVe, self).__init__(gensim_path, dim, binary, **kwargs)


class SpacyGloVe(WordEmbedding):
    def __init__(self):
        super(GloVe, self).__init__(300)
        self.nlp = spacy.load('en_vectors_web_lg')

    def get(self, word):
        return self.nlp.vocab.get_vector(word)


def get(embedding_name: str, *args, **kwargs) -> WordEmbedding:
    return globals()[embedding_name](*args, **kwargs)
