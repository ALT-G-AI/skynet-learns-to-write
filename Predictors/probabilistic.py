from collections import Counter
from string import punctuation
from numpy import log
from sklearn.base import BaseEstimator, ClassifierMixin
from data.data_examination import make_sig_words
from data.pipelines import (tokenize_pipe,
                            lower_pipe,
                            stem_pipe,
                            lemmatize_pipe)


class ProbabilisticClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            logTable={},
            counterTable={},
            beta_method=False,
            beta_stem=False,
            beta_lemma=False):
        """
        Called when initializing the classifier
        """
        self.counterTable = counterTable
        self.logTable = logTable
        self.beta_method = beta_method
        self.beta_stem = beta_stem
        self.beta_lemma = beta_lemma

    def fit(self, sentences, labels):
        distinct_labels = set(labels)

        if self.beta_method:
            for l in distinct_labels:
                self.logTable[l] = {}

            s_by_a = {a:
                      [s for s, a1 in zip(sentences, labels) if a1 == a]
                      for a in distinct_labels}

            tok_s_by_a = {
                k:
                list(tokenize_pipe(lower_pipe(v))) for k, v in s_by_a.items()}
            beta_table = make_sig_words(
                stem=self.beta_stem,
                lemma=self.beta_lemma,
                other_data=tok_s_by_a)

            self.beta_table = beta_table

            for l in beta_table:
                for w in beta_table[l]:
                    self.logTable[l][w] = log(beta_table[l][w])

            self.miss_p = {}
            for l in distinct_labels:
                self.miss_p[l] = min(self.logTable[l].values())

            self.trained_ = True
            return

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

    def score_(self, w, l):
        if self.hit_(w, l):
            return self.logTable[l][w]
        else:
            if self.beta_method:
                return self.miss_p[l]
            else:
                return 0

    def hit_(self, w, l):
        return (w in self.logTable[l])

    def predict(self, X):
        try:
            getattr(self, 'trained_')
        except AttributeError:
            raise RuntimeError('You must train the classifier before using it')
        X = lower_pipe(X)
        X = tokenize_pipe(X)
        if self.beta_stem:
            X = stem_pipe(X)
        if self.beta_lemma:
            X = lemmatize_pipe(X)

        X = list(X)

        return [self.predict_sen_(s) for s in X]

    def predict_sen_(self, s):
        words = s
        scores = [
            sum([self.score_(w, l) for w in words])
            for l in self.logTable.keys()]
        hits = [
            sum([self.hit_(w, l) for w in words])
            for l in self.logTable.keys()]

        if self.beta_method:
            maxin = scores.index(max(scores))
            return list(self.logTable.keys())[maxin]

        maxhits = max(hits)

        merged = zip(scores, hits, self.logTable.keys())

        maxes = [(s, l) for s, h, l in merged if h is maxhits]
        return max(maxes, key=lambda x: x[0])[1]
