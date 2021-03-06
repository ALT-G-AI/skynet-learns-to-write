import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

TRAINING_PATH = 'data/train.csv'


def import_data(training_path=TRAINING_PATH, languages=None):
    frame = pd.read_csv(training_path)

    train_set, test_set = train_test_split(
        frame,
        test_size=0.2,
        random_state=0)

    if languages is None:
        languages = []
    language_frames = [train_set]
    for language in languages:
        lang_frame = pd.read_csv('data/extended_data/train_{0}.csv'.format(language))
        lang_concat = [test_set, test_set, lang_frame]
        lang_frame = pd.concat(lang_concat)
        lang_frame.drop_duplicates(subset="id", inplace=True, keep=False)
        language_frames.append(lang_frame)

    train_set = pd.concat(language_frames)

    return train_set, test_set


def clean_data(word_set, remove_stopwords=False, lemmatize=False):
    if remove_stopwords:
        stopwords = nltk.corpus.stopwords.words('english')

    if lemmatize:
        lemm = WordNetLemmatizer()

    for index, row in word_set.iterrows():
        text_list = nltk.word_tokenize(word_set.text[index])

        if remove_stopwords:
            text_list = [word for word in text_list if word.lower() not in stopwords]

        if lemmatize:
            text_list = [lemm.lemmatize(word) for word in text_list]

        new_sentence = " ".join(text_list)

        word_set.text[index] = new_sentence

    return word_set


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
