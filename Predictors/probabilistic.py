from collections import Counter
from string import punctuation
from numpy import log
from data.import_data import import_data
from sklearn.model_selection import cross_val_score
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
            s = ''.join([c for c in s if c not in punctuation]).lower()
            words = s.split(' ')
            for w in words:
                self.counterTable[l][w] += 1

        for l in distinct_labels:
            ctr = self.counterTable[l]
            tw = sum(ctr.values())
            self.logTable[l] = {k: log(v / tw) for k, v in ctr.items()}
