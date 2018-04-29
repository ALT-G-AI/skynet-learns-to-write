from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, ClassifierMixin
from data.import_data import create_batched_ds, tokenize

import tensorflow as tf


class windowedGClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        encoder_size=100,
        window=5,
        DNNlayers=[100, 80],
        batch_n=1000,
        training_steps=100000
    ):

        self.encoder_size = encoder_size
        self.window = window
        self.DNNlayers = DNNlayers
        self.batch_n = batch_n
        self.training_steps = training_steps

    def fit(self, sentences, labels):

        print("Training word encoding")

        self.encoder = Word2Vec([
            tokenize(s)
            for s in sentences],
            size=self.encoder_size,
            window=self.window,
            min_count=0,
            workers=4)

        print("Training DNN")

        enc_size = self.encoder_size

        fc = tf.feature_column.numeric_column(
            'windows',
            shape=[self.window, enc_size])

        dnn_clf = tf.estimator.DNNClassifier(
            hidden_units=self.DNNlayers,
            n_classes=3,
            feature_columns=[fc])

        def input_fn():
            ds = create_batched_ds(
                self.encoder.wv,
                self.window,
                sentences,
                labels)

            self.author_key = ds[1]
            ds = ds[0]
            return ds.shuffle(1000).repeat().batch(self.batch_n)

        # logging.getLogger().setLevel(logging.INFO)
        dnn_clf.train(input_fn, steps=self.training_steps)

    def predict(self, X):
        pass
