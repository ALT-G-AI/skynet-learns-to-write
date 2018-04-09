from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, ClassifierMixin
from data.import_data import import_data
from sklearn.ensemble import RandomForestClassifier

import numpy as np
# import tensorflow as tf


class windowedForestClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        encoder_size=100,
        window=5,
    ):

        self.encoder_size = encoder_size
        self.window = window

    @staticmethod
    def tokenize_(sen):
        return word_tokenize(sen.lower())

    def windowed_inp_data_(self, sentences, labels, wlen):
        data = []
        outlabels = []

        for s, l in zip(sentences, labels):
            tokens = self.tokenize_(s)
            encs = [self.encoder[t] for t in tokens]

            for i in range(len(encs) + 1 - wlen):
                data.append(encs[i:i + wlen])
                outlabels.append(l)

        data = np.array(data)
        # sklearn.ensemble.RandomForestClassifier doesn't know what to do with
        # 3D data
        data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

        authors = set(outlabels)
        self.key = {k: v for k, v in zip(authors, range(len(authors)))}

        trans = [self.key[l] for l in outlabels]

        return data, trans

    def fit(self, sentences, labels):

        print("Training word encoding")

        self.encoder = Word2Vec([
            self.tokenize_(s)
            for s in sentences],
            size=self.encoder_size,
            window=self.window,
            min_count=0,
            workers=4)

        print("Training Forest")

        enc_size = self.encoder_size

        data, labels = self.windowed_inp_data_(sentences, labels, self.window)

        self.forest_clf = RandomForestClassifier(
            n_estimators=500,
            max_leaf_nodes=16,
            n_jobs=-1)

        self.forest_clf.fit(data, labels)

    def predict(self, X):
        tokens = self.tokenize_(X)
        encs = [self.encoder[t] for t in tokens]
        data = []
        for i in range(len(encs) + 1 - self.window):
            data.append(encs[i:i + self.window])

        data = np.array(data)
        data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

        reverse_key = {k: v for k, v in zip(
            self.key.values(), self.key.keys())}

        return [reverse_key[c] for c in self.forest_clf.predict(data)]
