import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

TRAINING_PATH = './data/train.csv'


def import_data(training_path=TRAINING_PATH):
    frame = pd.read_csv(training_path)
    train_set, test_set = train_test_split(
        frame,
        test_size=0.2,
        random_state=0)

    return train_set, test_set


def tokenize(sen):
    return word_tokenize(sen.lower())


def create_batched_ds(encoder, window, sens, labs):
    """
    batches datasets by author
    :param encoder:
    :param window:
    :param sens:
    :param labs:
    :return: (Full dataset
    """
    sentences = np.array(sens)
    authors = np.array(labs)

    unique_authors = list(set(authors))

    print("Key for authors is:\n", unique_authors)

    tokens = [tokenize(s) for s in sentences]

    encoded = np.array([np.array([encoder[w] for w in s]) for s in tokens])

    features = []
    author_labels = []

    for s, a in zip(encoded, authors):
        for i in range(len(s) + 1 - window):
            features.append(s[i:i + window])
            author_labels.append(a)

    features = np.array(features)
    author_labels = np.array(author_labels)

    dataset_full = tf.data.Dataset.from_tensor_slices(
        ({'windows': features}, author_labels))

    return dataset_full, {k: i for i, k in enumerate(unique_authors)}
