from collections import Counter

import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from numpy import log
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (confusion_matrix,
                             log_loss,
                             accuracy_score)
from sklearn.model_selection import cross_val_predict

from data.data_examination import make_sig_words
from data.import_data import import_data
from data.pipelines import (tokenize_pipe,
                            lower_pipe,
                            stem_pipe,
                            lemmatize_pipe)

from Predictors.probabilistic_NN import ProbabilisticNNClassifier
from Predictors.windowed_DNN import WindowedDNN


class GensembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            layers=[],
            epochs=150,
            index_out=True):
        """
        Called when initializing the classifier
        """
        self.layers = layers
        self.epochs = epochs
        self.index_out = index_out

    def fit(self, sentences, labels):

        listsen = list(sentences)
        listlab = list(labels)

        split=int(len(listsen * 0.9))

        b1s = listsen[:split]
        b1l = listlab[:split]
        b2s = listsen[split:]
        b2l = listsen[split:]

        self.pNN = ProbabilisticNNClassifier(
                    epochs=1,
                    layers=[],
                    beta_method=True,
                    stem=True,
                    lemma=True,
                    index_out=False)

        self. wNN = WindowedDNN(
                    epochs=100,
                    layers=[60],
                    window=10,
                    pte=True,
                    index_out=False)

        print("Fitting PNN")
        self.pNN.fit(b1s, b1l)

        print("Fitting WNN")
        self.wNN.fit(b1s, b1l)

        pNNpreds = np.array(self.pNN.predict(b2s))
        wNNpreds = np.array(self.wNN.predict(b2s))

        print("PNNPREDS", pNNpreds[:5])
        print("WNNPREDS", wNNpreds[:5])

        model = Sequential()

        if len(self.layers) > 0:
            fl = self.layers[0]

            model.add(Dense(fl, input_dim=6, activation='relu'))

            for l in self.layers[1:]:
                model.add(Dense(l, activation='relu'))

            model.add(Dense(3, activation='sigmoid'))
        else:
            model.add(Dense(3, input_dim=6, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        self.model = model

        model.summary()

        y_inp = to_categorical(b2l)

        model.fit(
            np.concatenate((pNNpreds, wNNpreds), 1),
            y_inp,
            epochs=self.epochs,)

        self.trained_ = True

    def predict(self, x):
        try:
            getattr(self, 'trained_')
        except AttributeError:
            raise RuntimeError('You must train the classifier before using it')

        pNNpreds = np.array(self.pNN.predict(x))
        wNNpreds = np.array(self.wNN.predict(x))

        out = self.model.predict(np.concatenate((pNNpreds, wNNpreds), 1))

        if self.index_out:
            return np.argmax(out, 1)
        else:
            return out


myc = []

if __name__ == '__main__':
    tr, te = import_data()

    author_enum = {'EAP': 0, 'HPL': 1, 'MWS': 2}

    classed_auths = [author_enum[a] for a in tr.author]

    myc = GensembleClassifier(
        epochs=100,
        layers=[4],
        index_out=False)

    y_train_pred = cross_val_predict(
        myc,
        tr.text,
        classed_auths,
        cv=3,
        n_jobs=-1)

    indices = np.argmax(np.array(y_train_pred), 1)

    print("Loss:", log_loss(classed_auths, y_train_pred))
    print("Acc:", accuracy_score(classed_auths, indices))

    CM = confusion_matrix(
        classed_auths,
        indices)

    # Get prob dists across rows
    prob_CM = CM / CM.sum(axis=1, keepdims=True)
    print(CM)
    print(prob_CM)
