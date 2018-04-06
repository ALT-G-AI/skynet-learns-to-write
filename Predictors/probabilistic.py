from collections import Counter
from string import punctuation
from numpy import log
from sklearn.base import BaseEstimator, ClassifierMixin


class ProbabilisticClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, logTable={}, counterTable={}):
        """
        Called when initializing the classifier
        """
        self.counterTable = counterTable
        self.logTable = logTable

    def fit(self, sentences, labels):
        distinct_labels = set(labels)
        for l in distinct_labels:
            self.counterTable[l] = Counter()

        for s, l in zip(sentences, labels):
            # Strip punctuation
            words = self.sen2words_(s)
            for w in words:
                self.counterTable[l][w] += 1

        for l in distinct_labels:
            ctr = self.counterTable[l]
            tw = sum(ctr.values())
            self.logTable[l] = {k: log(v / tw) for k, v in ctr.items()}

        self.trained_ = True

    @staticmethod
    def sen2words_(s):
        s = ''.join([c for c in s if c not in punctuation]).lower()
        return s.split(' ')

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

        return [self.predict_sen_(s) for s in X]

    def predict_sen_(self, s):
        words = self.sen2words_(s)
        scores = [
            sum([self.score_(w, l) for w in words])
            for l in self.logTable.keys()]
        hits = [
            sum([self.hit_(w, l) for w in words])
            for l in self.logTable.keys()]

        maxhits = max(hits)

        merged = zip(scores, hits, self.logTable.keys())

        maxes = [(s, l) for s, h, l in merged if h is maxhits]
        return max(maxes, key=lambda x: x[0])[1]
