from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, ClassifierMixin
from Processing.processing import merge_uncommon_words
from Predictors.windowed_data import windowed_data, predictor_data
import numpy as np
import tensorflow as tf


class windowedGClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        encoder_size=100,
        window=5,
        DNNlayers=[100, 80],
        batch_n=1000,
        training_steps=100000,
        min_word_thresh=3
    ):

        self.encoder_size = encoder_size
        self.window = window
        self.DNNlayers = DNNlayers
        self.batch_n = batch_n
        self.training_steps = training_steps
        self.min_word_thresh = min_word_thresh

    def fit(self, sentences, labels):

        print("Training word encoding")

        sentences = merge_uncommon_words(sentences, self.min_word_thresh)

        self.encoder = Word2Vec(
            sentences,
            size=self.encoder_size,
            window=self.window,
            min_count=0,
            workers=4)

        print("Training DNN")

        enc_size = self.encoder_size

        tf.logging.set_verbosity(tf.logging.FATAL)

        fc = tf.feature_column.numeric_column(
            'windows',
            shape=[self.window, enc_size])

        self.dnn_clf = tf.estimator.DNNClassifier(
            hidden_units=self.DNNlayers,
            n_classes=3,
            feature_columns=[fc],
            config=tf.estimator.RunConfig().replace(
                save_summary_steps=self.training_steps))

        input_fn = windowed_data(
            sentences,
            labels,
            self.window,
            self.batch_n,
            self.encoder)

        self.key = input_fn.key

        batch_count = 1

        while True:
            try:
                print("Running batch", batch_count)
                self.dnn_clf.train(
                    input_fn,
                    steps=self.training_steps)

                batch_count += 1
            except StopIteration:
                print("Finished Learning")
                break

    def predict(self, X):
        output_vector = []
        pred_count = 0
        for sen in X:
            print("Predicting output", pred_count)
            pred_count += 1

            def input_fn():
                return predictor_data(sen, self.encoder, self.window)

            gen = self.dnn_clf.predict(input_fn)
            preds = next(gen)['probabilities']

            sumprobs = np.sum([np.log(p) for p in preds], 0)

            index = np.argmax(sumprobs)

            inv_key = {v: k for k, v in self.key.items()}

            output_vector.append(inv_key[index])

        return output_vector
