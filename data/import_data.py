import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

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
    text = np.array(sens)
    authors = np.array(labs)

    author_set = set(authors)
    author_key = {k: v for k, v in zip(author_set, range(len(author_set)))}

    print("Key for authors is:\n", author_key)

    authors = np.array([author_key[a] for a in authors])

    tokens = [tokenize(s) for s in text]

    if type(encoder) is Word2Vec:
        encoder = encoder.wv

    encoded = np.array([np.array([encoder[w] for w in s]) for s in tokens])

    ext_encoded = []
    ext_authors = []

    for s, a in zip(encoded, authors):
        for i in range(len(s) + 1 - window):
            ext_encoded.append(s[i:i + window])
            ext_authors.append(a)

    ext_encoded = np.array(ext_encoded)
    ext_authors = np.array(ext_authors)

    dataset_full = tf.data.Dataset.from_tensor_slices(
        ({'windows': ext_encoded}, ext_authors))

    return dataset_full, author_key


STOPWORDS = set(stopwords.words('english'))
EXCLUDE = set(string.punctuation)
LEMMA = WordNetLemmatizer()


def clean(doc):
    """
        Cleans documents in preparation for topic learning
        - Removes stop words
        - Removes punctuation
        - Lemmatizes (groups similar words)
    """
    stop_free = " ".join([i for i in doc.lower().split() if i not in STOPWORDS])
    punc_free = ''.join(ch for ch in stop_free if ch not in EXCLUDE)
    normalized = " ".join(LEMMA.lemmatize(word) for word in punc_free.split())
    return normalized
