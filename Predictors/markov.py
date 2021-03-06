from collections import Counter

from numpy import log
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from data.import_data import import_data
from data.pipelines import (tokenize_pipe,
                            lower_pipe,
                            stem_pipe,
                            lemmatize_pipe)


class MarkovClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            stem=False,
            lemma=False):
        """
        Called when initializing the classifier
        """
        self.stem = stem
        self.lemma = lemma

    def pipeline_factory(self, sens):
        p = lower_pipe(sens)
        p = tokenize_pipe(p)
        if self.stem:
            p = stem_pipe(p)
        if self.lemma:
            p = lemmatize_pipe(p)
        return p

    def fit(self, sentences, labels):
        self.vocab = set()
        self.trans = {}
        self.logTable = {}

        distinct_labels = set(labels)

        self.labels = distinct_labels

        for l in distinct_labels:
            self.trans[l] = dict()

        sens = list(self.pipeline_factory(sentences))

        for s, l in zip(sens, labels):
            pairings = zip(s, s[1:])

            for w1, w2 in pairings:
                if w1 not in self.trans[l]:
                    self.trans[l][w1] = Counter()

                self.trans[l][w1][w2] += 1

        self.trained_ = True

    def _sen_prob(self, s, l):
        tokens = list(self.pipeline_factory([s]))[0]
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
                        sum([sum(v.values()) for v in trans.values()]))
            prob += log(wordprob)

        for w1, w2 in pairings:
            if w1 not in trans:
                miss_w += 1

            elif w2 not in trans[w1]:
                if w2 in trans:
                    wordprob = (sum(trans[w2].values()) /
                                sum([sum(v.values()) for v in trans.values()]))
                    prob += log(wordprob)
                miss_t += 1

            else:
                prob += log(trans[w1][w2] / sum(trans[w1].values()))

        return (prob, miss_w, miss_t)

    def _pred_sen(self, s):

        o_labels = list(self.labels)

        probs = [self._sen_prob(s, l) for l in o_labels]

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
        return o_labels[win_index]

    def predict(self, X):
        return [self._pred_sen(s) for s in X]


if __name__ == '__main__':
    tr, te = import_data()

    myc = MarkovClassifier(
        stem=True,
        lemma=True)
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
