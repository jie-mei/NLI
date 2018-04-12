import tensorflow as tf

from nn._base import Model
import data

if __name__ == '__main__':
    dataset = data.load('MSRP', 'train', 'Tokenize', 'Word2Vec', 256)
    model = Model(word_embeddings=dataset.embeds,
                  seq_len=dataset.max_len)
    print(model)
    print(type(tf.AUTO_REUSE))
    
