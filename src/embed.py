from abc import ABC, abstractmethod
import os
from typing import Dict, Set

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import spacy


class WordEmbedding(ABC):
    """
    Attributes:
        dim: The number of dimensions for each embedding vector.
    """
    def __init__(self, dim=0):
        self.dim = dim

    # TODO: Add default value as parameter
    @abstractmethod
    def get(self, word: str) -> np.ndarray:
        pass


class IndexedWordEmbedding(WordEmbedding):
    """
    Attributes:
        ids (Dict[str, int]): A mapping from word to id.
        embeds (np.array): A two dimentional array, where `embed[i]` stores the
            embedding for the word with id `i`.
    """

    def __init__(self, vocab: Set[str], embedding: WordEmbedding) -> None:
        super(IndexedWordEmbedding, self).__init__(embedding.dim)
        self.ids, self.embeds = {}, None  # type: Dict[str, int], np.ndarray
        if vocab:
            self.embeds = np.array([0] * embedding.dim)
            for wid, w in enumerate(vocab):
                self.ids[w] = wid
                self.embeds = np.vstack([self.embeds, embedding.get(w)])
            self.embeds = self.embeds[1:]

    def get_id(self, word: str) -> int:
        return self.ids[word]

    def get_embedding(self, wid: int) -> np.ndarray:
        return self.embeds[wid]

    def get(self, word: str) -> np.array:
        return self.get_embedding(self.get_id(word))


class PretrainedEmbedding(WordEmbedding):

    def __init__(self, path, dim, binary):
        super(PretrainedEmbedding, self).__init__(dim)
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path,
                binary=binary)
        self.unknowns = np.random.uniform(-0.1, 0.1, self.dim).astype("float32")

    def get(self, word):
        if word not in self.model.vocab:
            return self.unknowns
        else:
            return self.model.word_vec(word)


class Word2Vec(PretrainedEmbedding):
    def __init__(self,
                 path='/home/jmei/data/GoogleNews-vectors-negative300.bin',
                 dim=300,
                 binary=True):
        super(Word2Vec, self).__init__(path, dim, binary)


class GloVe(PretrainedEmbedding):
    def __init__(self,
                 path='/home/jmei/data/glove.840B.300d.txt',
                 dim=300,
                 binary=False):
        # Preprocess the original GloVe data to allow using the gensim API.
        gensim_path = '{}.gensim.txt'.format(path[:-4])
        if not os.path.exists(gensim_path):
            glove2word2vec(path, gensim_path)
        super(GloVe, self).__init__(gensim_path, dim, binary)


class SpacyGloVe(WordEmbedding):
    def __init__(self):
        super(GloVe, self).__init__(300)
        self.nlp = spacy.load('en_vectors_web_lg')

    def get(self, word):
        return self.nlp.vocab.get_vector(word)


def get(embedding_name: str, *args, **kwargs) -> WordEmbedding:
    return globals()[embedding_name](*args, **kwargs)
