from collections import Counter
from numpy import log
import numpy as np
from data.import_data import import_data
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from data.pipelines import (tokenize_pipe,
                            lower_pipe,
                            stem_pipe,
                            lemmatize_pipe,
                            strip_stopwords_pipe)


class ProbabilisticNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            logTable={},
            counterTable={},
            layers=[20],
            epochs=200,
            batch=500):
        """
        Called when initializing the classifier
        """
        self.counterTable = counterTable
        self.logTable = logTable
        self.layers = layers
        self.epochs = epochs
        self.batch = batch

    def fit(self, sentences, labels):
        print("Fitting")
        distinct_labels = set(labels)
        self.labels = distinct_labels
        for l in distinct_labels:
            self.counterTable[l] = Counter()

        print("Cleaning sentences")
        p1 = lower_pipe(sentences)
        p2 = tokenize_pipe(p1)
        p3 = stem_pipe(p2)
        p4 = lemmatize_pipe(p3)
        p4 = list(p4)
        sens = p4
        #sens = list(strip_stopwords_pipe(p4))

        print("Building probability data")
        for s, l in zip(sens, labels):
            # Strip punctuation
            for w in s:
                self.counterTable[l][w] += 1

        for l in distinct_labels:
            ctr = self.counterTable[l]
            tw = sum(ctr.values())
            self.logTable[l] = {k: log(v / tw) for k, v in ctr.items()}

        """ Feature list
            for each label -
                mean log prob 3
                std deviation log prob 3
                # of misses 3
                max log prob 3
                min log prob 3
                sentence length 1
        """

        print("Initialising NN")
        model = Sequential()
        if len(self.layers) > 0:
            fl = self.layers[0]

            model.add(Dense(fl, input_dim=16, activation='relu'))

            for l in self.layers[1:]:
                model.add(Dense(l, activation='relu'))

            model.add(Dense(3, activation='sigmoid'))
        else:
            model.add(Dense(3, input_dim=16, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        self.model = model

        model.summary()

        print("Fetching features")
        features = np.array([self.get_features(s) for s in sens])

        self.means = np.mean(features, 0)
        self.devs = np.std(features, 0)

        features = (features - self.means) / self.devs

        y_inp = to_categorical(labels)

        print("Training NN")
        model.fit(
            features,
            y_inp,
            epochs=self.epochs,
            batch_size=self.batch)

        self.model = model

        self.trained_ = True

    def get_features(self, s):
        wordprobs = [[self.score_(w, l) for w in s] for l in self.labels]
        wordhits = [[self.hit_(w, l) for w in s] for l in self.labels]

        means = np.mean(wordprobs, 1)
        devs = np.std(wordprobs, 1)
        miss = np.sum(wordhits, 1)
        maxs = np.max(wordprobs, 1)
        mins = np.min(wordprobs, 1)
        senlen = np.array([len(s)])

        return np.concatenate((means, devs, miss, maxs, mins, senlen))

    def score_(self, w, l):
        if self.hit_(w, l):
            return self.logTable[l][w]
        else:
            return 0

    def hit_(self, w, l):
        return (w in self.logTable[l])

    def predict(self, X):
        try:
            getattr(self, 'trained_')
        except AttributeError:
            raise RuntimeError('You must train the classifier before using it')

        p1 = lower_pipe(X)
        p2 = tokenize_pipe(p1)
        p3 = stem_pipe(p2)
        p4 = lemmatize_pipe(p3)
        p4 = list(p4)
        sens = p4
        #sens = list(strip_stopwords_pipe(p4))

        features = np.array([self.get_features(s) for s in sens])

        features = (features - self.means) / self.devs

        out = self.model.predict(features)

        return np.argmax(out, 1)


myc = []

if __name__ == '__main__':
    tr, te = import_data()

    author_enum = {'HPL': 0, 'EAP': 1, 'MWS': 2}

    classed_auths = [author_enum[a] for a in tr.author]

    myc = ProbabilisticNNClassifier(epochs=5000, layers=[])

    y_train_pred = cross_val_predict(
        myc,
        tr.text,
        classed_auths,
        cv=3,
        n_jobs=-1)

    CM = confusion_matrix(
        classed_auths,
        y_train_pred)

    #Get prob dists across rows
    prob_CM = CM / CM.sum(axis=1, keepdims=True)

    print(prob_CM)