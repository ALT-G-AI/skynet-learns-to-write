from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf


class windowedGClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        encoder=None,
        window=5,
        DNNlayers=['250,100']
    ):

        self.encoder = encoder
        self.window = window
        self.DNNlayers = DNNlayers

    @staticmethod
    def tokenize_(sen):
        return word_tokenize(sen.lower())

    def windowed_inp_data_(self, sentences, labels, wlen):
        data = {'windows': []}
        labels = []
        for s, l in zip(sentences, labels):
            tokens = self.tokenize_(s)
            encs = [self.encoder(t) for t in tokens]

            for i in range(len(encs) + 1 - wlen):
                data['windows'].append(encs[i:i + wlen])
                labels.append(l)

        return data, labels



    def fit(self, sentences, labels):

        print("Training word encoding")

        if self.encoder is None:
            self.encoder = Word2Vec([
                self.tokenize_(s)
                for s in sentences],
                size=250,
                window=self.window,
                min_count=1,
                workers=4)
            self.encoder.size = 250

        print("Training DNN")

        enc_size = self.encoder.size

        dnn_clf = tf.contrib.learn.DNNClassifier(
            hidden_units=self.DNNlayers,
            n_classes=3,
            feature_columns=)

        dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)

        def encoded_data():
            for s in sentences:
                return []

        return self.w2vEncoder

    def predict(self, X):
        pass
