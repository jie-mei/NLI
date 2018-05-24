import re
import typing as t

import numpy as np
import nltk
from nltk.corpus import wordnet

import base


# Feature indices
_TEMP = 0
_TAG  = 1
_MAT  = 2


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


def match(tokens1: t.List[str], tokens2: t.List[str]):
    """ Check whethe a word matches any word in the other sentence.

    Returns:
        Two lists of boolean values parallel with the input indicating whether
        words are matched.
    """
    words1 = map(lambda w: w.lower(), words1)
    words2 = map(lambda w: w.lower(), words2)
    stemmer = nltk.SnowballStemmer('english')
    stems1 = map(lambda w: stemmer.stem(w), words1)
    stems2 = map(lambda w: stemmer.stem(w), words2)
    def match_word(w1: str, s1: str, w2: str, s2: str):
        if w1 == w2 or s1 == s2:
            return True
        if (w1 == "n't" and w2 == 'not') or (w1 == 'not' and w2 == "n't"):
            return True
        for synsets in wordnet.synsets(w2):
            for lemma in synsets.lemma_names():
                if s1 == stemmer.stem(lemma):
                    return True
        return False
    match1 = [0] * len(tokens1)
    match2 = [0] * len(tokens2)
    for i1, (w1, s1) in enumerate(zip(words1, stems1)):
        for i2, (w2, s2) in enumerate(zip(words2, stems2)):
            if match_word(w1, s1, w2, s2):
                match1[i1], match2[i2] = 1, 1
                break
    return match1, match2



class SNLI(base.Dataset):
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

        # Collect all tag types in the dataset
        self.tags = {}  # type: t.Dict[str, int]
        for mode in ['train', 'validation', 'test']:
            for _, _, label, (_, tags1), (_, tags2) in self.parse(mode):
                for tag in tags1 + tags2:
                    if tag not in self.tags:
                        # Starts tag ID at 1 to differenciate with the zero
                        # paddings
                        self.tags[tag] = len(self.tags) + 1

        # Pad template to n-element-tuples
        for feats in self.x1_feats + self.x2_feats:
            for i in range(len(feats[0])):
                feats[_TEMP][i] = list(feats[_TEMP][i]) \
                        + [0] * (self.TEMP_DROP_VAL - len(feats[_TEMP][i]))
            if not feats[_TEMP]:
                # To aviod zero shape error in TF
                feats[_TEMP] = [[0] * self.TEMP_DROP_VAL]
            # Transform tags to tag IDs
            feats[_TAG] = [self.tags[t] for t in feats[_TAG]]

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
                            return words, intls, w_tags + i_tags
                        words1, temp1, tags1 = process(fields[3])
                        words2, temp2, tags2 = process(fields[4])
                        match1, match2 = match(words1, words2)
                        yield (' '.join(words1),        # sent1
                               ' '.join(words2),        # sent2
                               label,                   # label
                               [temp1, tags1, match1],  # feats1
                               [temp2, tags2, match2])  # feats2

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
