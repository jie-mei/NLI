from abc import ABC, abstractmethod
from typing import Dict, Set

import gensim
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


class Word2Vec(WordEmbedding):
    PRETRAINED_PATH = '~/data/GoogleNews-vectors-negative300.bin'

    def __init__(self, pretrained=PRETRAINED_PATH):
        super(Word2Vec, self).__init__(300)
        self.model = (gensim.models.KeyedVectors
                .load_word2vec_format(pretrained, binary=True))
        self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

    def get(self, word):
        if word not in self.model.vocab:
            return self.unknowns
        else:
            return self.model.word_vec(word)


class GloVe(WordEmbedding):
    def __init__(self):
        super(GloVe, self).__init__(300)
        self.nlp = spacy.load('en_vectors_web_lg')

    def get(self, word):
        return self.nlp.vocab.get_vector(word)


def get(embedding_name: str, *args, **kwargs) -> WordEmbedding:
    return globals()[embedding_name](*args, **kwargs)
