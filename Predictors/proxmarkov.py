import random
from collections import Counter
from difflib import get_close_matches

from gensim.models import KeyedVectors
from numpy import log, exp, power
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from data.import_data import tokenize, import_data


class ProxMarkovClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        """
        Called when initializing the classifier
        """
        self.w2v = KeyedVectors.load_word2vec_format(
            './data/glove/glove.6B.50d.txt',
            binary=False)

    def fit(self, sentences, labels):

        self.vocab = set()
        self.trans = {}
        self.logTable = {}

        distinct_labels = set(labels)

        self.labels = distinct_labels

        for l in distinct_labels:
            self.trans[l] = dict()
        count = 0
        for s, l in zip(sentences, labels):
            if count % 50 == 0:
                print("Training: ", count)
            count += 1

            tokens = tokenize(s, stop=False)
            pairings = zip(tokens, tokens[1:])

            for w1, w2 in pairings:
                w1 = self.fix_word(w1)
                w2 = self.fix_word(w2)
                if w1 not in self.trans[l]:
                    self.trans[l][w1] = Counter()

                self.trans[l][w1][w2] += 1

        self.runcount = 0

    def fix_word(self, w):
        vocab = self.w2v.wv.vocab
        if w in vocab:
            return w
        wn = self.failsafe_match(w, vocab, n=3)[0]

        self.trained_ = True
        self.falloff = 0.01
        return wn

    def compression(self, d):
        return exp(-self.falloff * power(d, 2)) + 1e-8

    def nearest_word(self, w, words):
        if w in words:
            return w, 1
        else:
            nw = self.w2v.wv.most_similar_to_given(w, words)
            sim = self.w2v.wv.distance(w, nw)

            return nw, self.compression(sim)

    @classmethod
    def failsafe_match(self, w, vocab, n):
        try:
            out = get_close_matches(w, vocab, n=n)[0]
        except IndexError:
            out = random.choice(vocab)
        return out

    def _sen_prob(self, s, l):
        if self.runcount % 25 == 0:
            print("RUNS:", self.runcount)
        self.runcount += 1
        tokens = tokenize(s, stop=False)
        pairings = zip(tokens, tokens[1:])

        trans = self.trans[l]

        prob = 0

        fw = tokens[0]
        fwp = 1
        wordprob = 0

        if fw in trans:
            wordprob = (sum(trans[fw].values()) /
                        sum([sum(v.values()) for v in trans.values()]))
        else:
            try:
                fw, fwp = self.nearest_word(fw, list(trans.keys()))
            except KeyError:
                fw = self.failsafe_match(fw, self.w2v.wv.vocab, n=1)[0]
                fw, fwp = self.nearest_word(fw, list(trans.keys()))

                wordprob = (sum(trans[fw].values()) /
                            sum([sum(v.values()) for v in trans.values()]))

        prob += log(wordprob) + log(fwp)

        for w1, w2 in pairings:

            try:
                w1p = 1
                trans2 = trans[w1]
            except KeyError:
                try:
                    w1, w1p = self.nearest_word(w1, list(trans.keys()))
                except KeyError:
                    w1 = self.failsafe_match(w1, self.w2v.wv.vocab, n=1)[0]
                    w1, w1p = self.nearest_word(w1, list(trans.keys()))
                trans2 = trans[w1]

            try:
                w2p = 1
                tr_p = trans2[w2]
            except KeyError:
                try:
                    w2, w2p = self.nearest_word(w2, list(trans2.keys()))
                except KeyError:
                    w2 = self.failsafe_match(w2, self.w2v.wv.vocab, n=1)[0]
                    w2, w2p = self.nearest_word(w2, list(trans.keys()))
                tr_p = trans2[w2]

            prob += log(tr_p / sum(trans2.values())) + \
                    log(w1p) + \
                    log(w2p)

        return prob

    def _pred_sen(self, s):

        self.runcount += 1
        if self.runcount % 25 == 0:
            print("RUNS:", self.runcount)

        o_labels = list(self.labels)

        probs = [self._sen_prob(s, l) for l in o_labels]

        winner = probs.index(max(probs))

        return o_labels[winner]

    def predict(self, X):
        return [self._pred_sen(s) for s in X]


if __name__ == '__main__':
    tr, te = import_data()

    myc = ProxMarkovClassifier()
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
