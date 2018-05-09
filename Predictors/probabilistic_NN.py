from collections import Counter

import numpy as np
from keras.layers import Dense
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


class ProbabilisticNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            log_table=None,
            counter_table=None,
            layers=None,
            epochs=200,
            batch=500,
            beta_method=False,
            stem=False,
            lemma=False,
            index_out=True):
        """
        Called when initializing the classifier
        """
        if layers is None:
            layers = [20]
        if log_table is None:
            log_table = {}
        if counter_table is None:
            counter_table = {}
        self.counterTable = counter_table
        self.logTable = log_table
        self.layers = layers
        self.epochs = epochs
        self.batch = batch
        self.beta_method = beta_method
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
        print("Fitting")
        distinct_labels = set(labels)
        self.labels = distinct_labels
        for l in distinct_labels:
            self.counterTable[l] = Counter()

        print("Cleaning sentences")
        sens = self.pipeline_factory(sentences)
        sens = list(sens)
        print("Building probability data")
        if not self.beta_method:
            for s, l in zip(sens, labels):
                # Strip punctuation
                for w in s:
                    self.counterTable[l][w] += 1

            for l in distinct_labels:
                ctr = self.counterTable[l]
                tw = sum(ctr.values())
                self.logTable[l] = {k: log(v / tw) for k, v in ctr.items()}

        else:
            if self.beta_method:
                for l in distinct_labels:
                    self.logTable[l] = {}

                s_by_a = {a:
                              [s for s, a1 in zip(sentences, labels) if a1 == a]
                          for a in distinct_labels}

                tok_s_by_a = {
                    k:
                        list(self.pipeline_factory(v)) for k, v in s_by_a.items()}
                beta_table = make_sig_words(
                    stem=self.stem,
                    lemma=self.lemma,
                    other_data=tok_s_by_a)

                self.beta_table = beta_table

                for l in beta_table:
                    for w in beta_table[l]:
                        self.logTable[l][w] = log(beta_table[l][w])

                self.miss_p = {}
                for l in distinct_labels:
                    self.miss_p[l] = min(self.logTable[l].values())

        # Feature list
        #     for each label -
        #         mean log prob 3
        #         std deviation log prob 3
        #         # of misses 3
        #         max log prob 3
        #         min log prob 3
        #         sentence length 1

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
        miss = np.sum(wordhits, 1) / len(s)
        maxs = np.max(wordprobs, 1)
        mins = np.min(wordprobs, 1)
        senlen = np.array([len(s)])

        return np.concatenate((means, devs, miss, maxs, mins, senlen))

    def score_(self, w, l):
        if self.hit_(w, l):
            return self.logTable[l][w]
        else:
            if self.beta_method:
                return self.miss_p[l]
            else:
                return 0

    def hit_(self, w, l):
        return w in self.logTable[l]

    def predict(self, x):
        try:
            getattr(self, 'trained_')
        except AttributeError:
            raise RuntimeError('You must train the classifier before using it')

        sens = self.pipeline_factory(x)
        sens = list(sens)

        features = np.array([self.get_features(s) for s in sens])

        features = (features - self.means) / self.devs

        out = self.model.predict(features)

        if self.index_out:
            return np.argmax(out, 1)
        else:
            return out


myc = []

if __name__ == '__main__':
    tr, te = import_data()

    author_enum = {'HPL': 0, 'EAP': 1, 'MWS': 2}

    classed_auths = [author_enum[a] for a in tr.author]

    myc = ProbabilisticNNClassifier(
        epochs=1500,
        layers=[],
        beta_method=True,
        stem=True,
        lemma=True,
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

    print(prob_CM)
