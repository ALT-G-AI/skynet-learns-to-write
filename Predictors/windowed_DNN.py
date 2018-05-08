from collections import Counter
from numpy import log
from sklearn.base import BaseEstimator, ClassifierMixin
from data.import_data import tokenize, import_data

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from gensim.models import Word2Vec

from data.glove.pre_encoder import pte

from data.pipelines import (tokenize_pipe,
                            lower_pipe,
                            stem_pipe,
                            lemmatize_pipe,
                            uncommon_pipe,
                            encode_pipe,
                            window_pipe,
                            window_pipe_nolabel,
                            cull_words_pipe)

import numpy as np


class windowedDNN(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            window=5,
            layers=[50, 25],
            word_dim=50,
            epochs=250,
            batch=1000,
            verbose=True,
            pte=False,
            stem=False,
            lemma=False,
            index_out=True):
        """
        Called when initializing the classifier
        """
        self.window = window
        self.layers = layers
        self.word_dim = word_dim
        self.epochs = epochs
        self.batch = batch
        self.verbose = verbose
        self.pte = pte
        self.stem = stem
        self.lemma = lemma
        self.index_out = index_out

    def pipeline_factory(self, sens):
        p = lower_pipe(sens)
        p = tokenize_pipe(p)
        if self.stem:
            p = stem_pipe(p)
        if self.lemma:
            p = lemmatize_pipe(p)
        return p

    def fit(self, sentences, labels):

        if self.verbose:
            print("Building NN")
        model = Sequential()

        model.add(Flatten(input_shape=(self.window, self.word_dim)))

        for l in self.layers:
            model.add(Dense(l, activation='relu'))

        model.add(Dense(3, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        self.model = model

        if self.verbose:
            model.summary()

        pipe_out = self.pipeline_factory(sentences)

        if not self.pte:
            pipe_out = uncommon_pipe(pipe_out)

        clean_sens = list(pipe_out)
        self.clean_sens = clean_sens

        if self.verbose:
            print("Building word embedding")

        self.encoder = None
        if self.pte:
            self.encoder = pte()
        else:
            self.encoder = Word2Vec(
                clean_sens,
                size=self.word_dim,
                min_count=0)

        enc = encode_pipe(clean_sens, self.encoder)
        windows = list(window_pipe(enc, labels, self.window))

        win_sens = [w[0] for w in windows]
        win_labs = [w[1] for w in windows]

        y_inp = to_categorical(win_labs)

        if self.verbose:
            print("Training NN")

        model.fit(
            np.array(win_sens),
            np.array(y_inp),
            epochs=self.epochs,
            batch_size=self.batch,
            verbose=self.verbose)

    def _pred_sen(self, s):
        s_array = [s]

        pipe_out = self.pipeline_factory(s_array)

        if not self.pte:
            pipe_out = cull_words_pipe(pipe_out, self.encoder.wv.vocab)

        pipe_out = encode_pipe(pipe_out, self.encoder)
        windows = np.array(list(window_pipe_nolabel(pipe_out, self.window)))

        preds = self.model.predict(windows, batch_size=len(windows))

        logs = np.log(preds)
        flat = np.sum(logs, 0)

        if self.index_out:
            winner_index = np.argmax(flat)
            return winner_index
        else:
            return flat / sum(flat)

    def predict(self, X):
        return [self._pred_sen(s) for s in X]


myc = []

if __name__ == '__main__':

    tr, te = import_data()
    author_enum = {'HPL': 0, 'EAP': 1, 'MWS': 2}

    classed_auths = [author_enum[a] for a in tr.author]

    myc = windowedDNN(
        layers=[100],
        window=8,
        pte=False,
        verbose=True,
        epochs=150,
        index_out=False)

    y_train_pred = cross_val_predict(
        myc,
        tr.text,
        classed_auths,
        cv=3,
        n_jobs=-1)

    logloss = log_loss(classed_auths, y_train_pred)

    print("Loss:", logloss)

    indexes = np.argmax(np.array(y_train_pred), 1)

    CM = confusion_matrix(
        classed_auths,
        indexes)

    prob_CM = CM / CM.sum(axis=1, keepdims=True)

    print(prob_CM)

    acc = accuracy_score(classed_auths, indexes)
    print("Acc:", acc)
    print("----------------------------------------\n")
