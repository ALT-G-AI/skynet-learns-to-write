from collections import Counter
from numpy import log
from sklearn.base import BaseEstimator, ClassifierMixin
from import_data import tokenize


class Markov(BaseEstimator, ClassifierMixin):
    def __init__(self):
        """
        Called when initializing the classifier
        """

    def fit(self, sentences, labels):
        self.vocab = set()
        self.trans = {}
        self.logTable = {}

        distinct_labels = set(labels)

        self.labels = distinct_labels

        for l in distinct_labels:
            self.trans[l] = dict()

        for s, l in zip(sentences, labels):
            tokens = tokenize(s, stop=True)
            pairings = zip(tokens, tokens[1:])

            for w1, w2 in pairings:
                if w1 not in self.trans[l]:
                    self.trans[l][w1] = Counter()

                self.trans[l][w1][w2] += 1

        self.trained_ = True

    def _sen_prob(self, s, l):
        tokens = tokenize(s, stop=True)
        pairings = zip(tokens, tokens[1:])

        trans = self.trans[l]

        miss_w = 0
        miss_t = 0
        prob = 0

        first_word = tokens[0]

        if first_word not in trans:
                miss_w += 1
        else:
            wordprob = (sum(trans[first_word].values()) /
                        sum([sum(w.values()) for w in trans]))
            prob += log(wordprob)

        for w1, w2 in pairings:
            if w1 not in trans:
                miss_w += 1

            elif w2 not in trans[w1]:
                wordprob = (sum(trans[w2].values()) /
                            sum([sum(w.values()) for w in trans]))
                prob += log(wordprob)
                miss_t += 1

            else:
                prob += log(trans[w1][w2] / sum(trans[w1].values()))

        return (prob, miss_w, miss_t)

    def _pred_sen(self, s):
        probs = [self._sen_prob(s, l) for l in self.labels]

        # Get minimum missed words
        minw = min([p[1] for p in probs])

        # Get all labels with min missed words
        winners_w = [p for p in probs if p[1] == minw]

        # Of those, get minimum missed transitions
        mint = min([p[2] for p in winners_w])

        # Get all labels with min missed transitions
        winners_t = [p for p in winners_w if p[2] == mint]

        # Get the highest prob label
        winner = max(winners_t, key=lambda x: x[0])

        # Get its index
        win_index = probs.index(winner)

        return self.labels[win_index]

    def predict(self, X):
        return [self._pred_sen(s) for s in X]
