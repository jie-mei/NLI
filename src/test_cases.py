import re

import data

def parse_tree(string):
    """ Parse the syntax tree given the sentence parse marker. This
    function returns two lists of tokens and crossponding POS tags,
    respectively. """
    tags, words = [], []
    for mo in re.finditer(r'\(([^\s()]+) ([^\s()]+)\)', string):
        tag, word = mo.group(1, 2)
        tags += tag,
        words += word,
    return tags, words


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


def test_parse():
    with open(data.SNLI.DATA_FILES['validation'][0], 'r', encoding='utf-8') as f:
        f.readline()  # skip the heading line
        for line in f:
            fields  = line.strip().split('\t')
            t1, t2 = fields[3:5]
            print(t1)
            print()
            print()
            tree = ParseTree.parse(t1)
            print(tree)
            print(tree.get_leafs())
            print(tree.get_internals())
            break

def test_parse_example():
    exp = "(ROOT (NP (NP (CD Three) (NNS people)) (, ,) (NP (NP (CD one) (JJ tall) (NN male)) (PP (IN in) (NP (NP (JJ blue) (NN polo) (NN shirt) (CC and) (NN khaki) (NNS shorts)) (PP (IN with) (NP (JJ tan) (NNS shoes)))))) (, ,) (NP (NN brunette)) (, ,) (NP (NP (JJ heavy) (JJ chested) (NN female)) (PP (IN in) (NP (JJ white) (NN summer) (NN blouse) (CC and) (NN denim) (NNS shorts)))) (, ,) (S (NP (JJ second) (NN female)) (VP (VBG wearing) (NP (JJ yellow) (NN rimmed) (NNS sunglasses)))) (, ,) (NP (NP (JJ yellow) (NN blouse)) (CC and) (NP (NNS sandals))) (, ,) (VP (VBG walking) (PP (IN across) (NP (DT a) (NN city) (NN street)))) (. .)))"
    tree = ParseTree.parse(exp)
    def preorder(t):
        if t:
            print(t.word, t.tag, len(t.children))
            for c in t.children:
                preorder(c)
    print(exp)
    preorder(tree)
    print(tree.get_internals())
    print(tree.get_internals(4))
#test_parse_example()

def test_max_internal_child():
    for s1, s2, label, feat1, feat2 in data.SNLI.parse('validation'):
        break

def test_data_shape():
    dset = data.load_dataset('SNLI', 'train', 'GloVeNorm', 6523)
    max_len = 3
    for i in range(len(dset.x1_feats)):
        for val in dset.x1_feats[i][0] + dset.x2_feats[i][0]:
            if len(val) > max_len:
                max_len = len(val)
                print(i)
                print(dset.x1_words[i])
                print(dset.x1_words[i])
                print(dset.x1_feats[i])
                print(dset.x1_feats[i])
                print()
    print(max_len)
#test_data_shape()

def test_temp_shape_dist():
    import collections
    dset = data.load_dataset('SNLI', 'train', 'GloVeNorm', 6523)
    len_dist = collections.defaultdict(int)
    for i in range(len(dset.x1_feats)):
        for val in dset.x1_feats[i][0] + dset.x2_feats[i][0]:
            len_dist[len(val)] += 1
    total = sum(len_dist.values())
    len_dist = {k: v / total for k, v in len_dist.items()}
    print(len_dist)
#test_temp_shape_dist()

def test_temp_len_dist():
    import collections
    dset = data.load_dataset('SNLI', 'train', 'GloVeNorm', 6523)
    len_dist = collections.defaultdict(int)
    for i in range(len(dset.x1_feats)):
        len_dist[len(dset.x1_feats[i][0])] += 1
        len_dist[len(dset.x2_feats[i][0])] += 1
        if not len(dset.x1_feats[i][0]) or not len(dset.x2_feats[i][0]):
            print(i)
            print(dset.x1_words[i])
            print(dset.x1_feats[i])
            print(dset.x2_words[i])
            print(dset.x2_feats[i])
            print()
    total = sum(len_dist.values())
    len_dist = {k: v / total for k, v in len_dist.items()}
    #for k in sorted(len_dist.keys()):
    #    print(k, len_dist[k])
test_temp_len_dist()
