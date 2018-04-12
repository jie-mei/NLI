from abc import ABC, abstractmethod
from typing import List
import string

from util.display import ReprMixin


# Hardcoded CoreNLP stopwords.
STOPWORDS = set([
        'a', 'an', 'the', 'of', 'at', 'on', 'upon', 'in', 'to', 'from', 'out',
        'as', 'so', 'such', 'or', 'and', 'those', 'this', 'these', 'that',
        'for', 'is', 'was', 'am', 'are', '\'s', 'been', 'were',
        ])


def _tokenize(text):
    return nltk.word_tokenize(text)


def _numeric_features(tokens1, tokens2):
    """ Return two binary values indicating whether two tokenized inputs:
        * contains the same numeric tokens
        * contains overlapping numeric tokens
    """
    def find_numerics(tokens):
        return set(tk for tk in tokens if any(c in string.digit for c in tk))
    nums1, nums2 = find_numerics(tokens1), find_numerics(tokens2)
    return bool(nums1 == nums2), bool(nums1 & nums2)


class DataPreproc(ABC):
    def __init__(self, steps: List['DataPreproc'] = []) -> None:
        self.steps = steps

    @property
    def __abbr__(self):
        return '_'.join(s.__abbr__ for s in self.steps)

    def preproc(self, tokens: List[str]) -> List[str]:
        for step in self.steps:
            print(step.__class__.__name__)
            tokens = step.preproc(tokens)
        return tokens


class Tokenize(DataPreproc):
    def __abbr__(self):
        return 'tkz'


class RemoveStopWords(DataPreproc):
    def __abbr__(self):
        return 'sw'

    def preproc(self, tokens: List[str]) -> List[str]:
        return ['' if w.lower() in STOPWORDS else w for w in tokens]


class RemovePunctuation(DataPreproc):
    def __abbr__(self):
        return 'punct'

    def preproc(self, tokens: List[str]) -> List[str]:
        return [w if any(c not in string.punctuation for c in w) else ''
                for w in tokens]


class ToLowerCase(DataPreproc):
    def __abbr__(self):
        return 'low'

    def preproc(self, tokens: List[str]) -> List[str]:
        return [w.lower() for w in tokens]


class DataClean(DataPreproc):
    def __init__(self):
        steps = [RemoveStopWords, RemovePunctuation, ToLowerCase]
        super(DataClean, self).__init__(steps)
