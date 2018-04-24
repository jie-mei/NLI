import pickle
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer


def fit_tokenizer(texts, num_words=None):
    tokenizer = Tokenizer(num_words, lower=False)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def tokenize_texts(X, tokenizer, max_len):
    """tokenize each report into a vector using the keras tokenizer"""
    if tokenizer is None:
        tokenizer = fit_tokenizer(X)
    sequences = tokenizer.texts_to_sequences(X)
    data = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return data
