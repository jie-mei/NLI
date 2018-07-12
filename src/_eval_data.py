import data
import numpy as np


LIMIT = 20


def _print_word_embed(dataset, word):
    embed = dataset.word_embedding
    wid = embed.get_id(word)
    wemb = embed.get(word)
    is_oov = embed.is_oov(word)
    size = np.linalg.norm(wemb)
    print('%10d %20s %s %4.2f %s' % (wid, word, 'OOV' if is_oov else '   ', size, wemb[:3]))


def print_embed(data_cls, mode, embed_cls, seed):
    dataset = data.load_dataset(data_cls, mode, embed_cls, seed)
    print(dataset.x1_words[0])

    embed = dataset.word_embedding
    for i, (word, wid) in enumerate(embed._IndexedWordEmbedding__ids.items()):
        if i < LIMIT:
            wemb = embed.get(word)
            is_oov = embed.is_oov(word)
            size = np.linalg.norm(wemb)
            print('%10d %20s %s %4.2f %s' % (wid, word, 'OOV' if is_oov else '   ', size, wemb[:3]))

    if '<EOS>' in embed._IndexedWordEmbedding__ids:
        _print_word_embed(dataset, '<EOS>')
    if '<BOS>' in embed._IndexedWordEmbedding__ids:
        _print_word_embed(dataset, '<BOS>')


def print_tag(data_cls, mode, embed_cls, seed):
    dataset = data.load_dataset(data_cls, mode, embed_cls, seed)
    for i in range(3):
        words = dataset.x2_words[i]
        tags = dataset.x2_feats[i][1]
        print([(w, t) for w, t in zip(words, tags)])
    


if __name__ == '__main__':
    # Print 
    print_tag('SNLI', 'test', 'GloVe', 6523)
    print()
    print_tag('SNLI', 'test', 'GloVeNorm', 6523)
    print()
    print_tag('SNLI', 'train', 'GloVeNorm', 6523)
    print()
    print()
    print()

