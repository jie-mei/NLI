import copy
from functools import reduce

import bs4
import numpy as np
import seaborn as sns


class AttentedText(object):
    def __init__(self, words, attents):
        self.words = words
        self.attents = attents


class TextSaliencyMap():
    def __init__(self):
        """
        :type source: collections.Iterable[IPipelineData]
        """

        self.__soup = bs4.BeautifulSoup('', 'html.parser')
        self.__body = self.__soup.new_tag('body')
        self.__soup.insert(0, self.__body)

    def add_colored_text(self, texts, prediction_class, true_class,
            doc_id=None, prediction_probability=None):
        # Colors
        clrs_pos = sns.color_palette("Blues", 512)  # 256 + 128 = 384
        clrs_neg = sns.color_palette("Reds", 512)
        max_clr = np.max(list(map(abs, reduce(lambda x, y: x.attents + y.attents, texts))))
        # Insert title
        title = self.__soup.new_tag('p', style="background-color: #FFFFFF")
        title.string = ("prediction_class: {}, true_class: {}"
                .format(prediction_class, true_class))
        if prediction_probability is not None:
            title.string += ('prediction_probability: {}'
                    .format(prediction_probability))
        if doc_id is not None:
            title.string += 'doc_id: {}'.format(doc_id)
        self.__body.insert(float("inf"), title)
        def insert_text(text):
            # Insert text attention values
            #att = self.__soup.new_tag('p', style="background-color: #FFFFFF")
            #att.string = str(text.attents)
            #self.__body.insert(float("inf"), att)
            # Insert colored texts
            weight_values = copy.deepcopy(text.attents)
            abs_weight_values = list(map(abs, text.attents))
            color_values = (abs_weight_values / max_clr * 255).astype(np.int32)
            for word, weight, value in zip(text.words, color_values, weight_values):
                if not word:
                    break
                if value >= 0:
                    color = tuple([int(255 * i) for i in clrs_pos[weight]][0:3])
                else:
                    color = tuple([int(255 * i) for i in clrs_neg[weight]][0:3])
                rgb_hex = "#%02x%02x%02x" % color
                if word == '\n':
                    colored_word = self.__soup.new_tag('br',
                            style="background-color: #FFFFFF")
                else:
                    colored_word = self.__soup.new_tag('span',
                            title="{:.4f}".format(value),
                            style="background-color: " + rgb_hex)
                    colored_word.string = word
                self.__body.insert(float("inf"), colored_word)
            self.__body.insert(float("inf"), self.__soup.new_tag('p'))
        for t in texts:
            insert_text(t)
        self.__body.insert(float("inf"), self.__soup.new_tag('hr'))

    def write(self, path):
        with open(path, 'w') as outfile:
            outfile.write(self.__soup.prettify())


if __name__ == '__main__':
    t1 = AttentedText(['hello', ',', 'world', '!'], [-5, 3, 10, -2])
    t2 = AttentedText(['hello', ',', 'god', '!'], [-5, 0, 4, -2])
    t = TextSaliencyMap()
    t.add_colored_text([t1, t2], 1, 1)
    t.write('testplot')
