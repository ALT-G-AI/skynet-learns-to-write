from collections import Counter

from numpy import log
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from data.import_data import import_data, tokenize


class ProbabilisticWithFillClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, log_table=None, counter_table=None, stem=False, cull_stopwords=False):
        """
        Called when initializing the classifier
        """
        if log_table is None:
            log_table = {}
        if counter_table is None:
            counter_table = {}
        self.counterTable = counter_table
        self.logTable = log_table
        self.stem = stem
        self.cull_stopwords = cull_stopwords

    def fit(self, sentences, labels):
        distinct_labels = set(labels)
        for l in distinct_labels:
            self.counterTable[l] = Counter()

        for s, l in zip(sentences, labels):
            # Strip punctuation
            words = tokenize(
                s,
                stem=self.stem,
                cull_stopwords=self.cull_stopwords)
            for w in words:
                self.counterTable[l][w] += 1

        for l in distinct_labels:
            ctr = self.counterTable[l]
            tw = sum(ctr.values())
            self.logTable[l] = {k: log(v / tw) for k, v in ctr.items()}

        self.trained_ = True

    def hit_(self, w, l):
        return w in self.logTable[l]

    def score_(self, w, l):
        if self.hit_(w, l):
            return self.logTable[l][w]
        else:
            return log(1 / sum(self.counterTable[l].values()))

    def score_sen_(self, s, l):
        return sum([self.score_(w, l) for w in s])

    def predict(self, X):
        try:
            getattr(self, 'trained_')
        except AttributeError:
            raise RuntimeError('You must train the classifier before using it')

        return [self.predict_sen_(s) for s in X]

    def predict_sen_(self, s):
        s = tokenize(
            s,
            stem=self.stem,
            cull_stopwords=self.cull_stopwords)
        scores = [(l, self.score_sen_(s, l)) for l in self.logTable.keys()]

        winner = max(scores, key=lambda x: x[1])
        return winner[0]


if __name__ == '__main__':
    tr, te = import_data()

    myc = ProbabilisticWithFillClassifier(cull_stopwords=True)
    y_train_pred = cross_val_predict(
        myc,
        tr.text,
        tr.author,
        cv=3,
        n_jobs=-1)

    CM = confusion_matrix(
        tr.author,
        y_train_pred,
        labels=["EAP", "HPL", "MWS"])

    # Get prob dists across rows
    prob_CM = CM / CM.sum(axis=1, keepdims=True)

    print(prob_CM)
