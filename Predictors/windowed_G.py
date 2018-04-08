from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import numpy as np
import logging


class windowedGClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        encoder=None,
        window=5,
        DNNlayers=[100, 80]
    ):

        self.encoder = encoder
        self.window = window
        self.DNNlayers = DNNlayers

    @staticmethod
    def tokenize_(sen):
        return word_tokenize(sen.lower())

    def windowed_inp_data_(self, sentences, labels, wlen):
        data = {'windows': []}
        outlabels = []
        for s, l in zip(sentences, labels):
            tokens = self.tokenize_(s)
            encs = [self.encoder[t] for t in tokens]

            for i in range(len(encs) + 1 - wlen):
                data['windows'].append(encs[i:i + wlen])
                outlabels.append(l)

        data['windows'] = np.array(data['windows'])

        authors = set(outlabels)
        key = {k: v for k, v in zip(authors, range(len(authors)))}

        trans = [key[l] for l in outlabels]

        print('Author to label encoding is:', key)

        return data, trans

    def fit(self, sentences, labels):

        print("Training word encoding")

        def_enc_size = 100

        if self.encoder is None:
            self.encoder = Word2Vec([
                self.tokenize_(s)
                for s in sentences],
                size=def_enc_size,
                window=self.window,
                min_count=0,
                workers=4)
            self.encoder.size = def_enc_size

        print("Training DNN")

        enc_size = self.encoder.size

        fc = tf.feature_column.numeric_column(
            'windows',
            shape=[self.window, enc_size])

        def input_fn():
            return self.windowed_inp_data_(sentences, labels, self.window)

        dnn_clf = tf.estimator.DNNClassifier(
            hidden_units=self.DNNlayers,
            n_classes=3,
            feature_columns=[fc])

        logging.getLogger().setLevel(logging.INFO)
        dnn_clf.train(input_fn)

        def encoded_data():
            for s in sentences:
                return []

        return self.w2vEncoder

    def predict(self, X):
        pass
