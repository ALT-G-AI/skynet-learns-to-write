from collections import Counter
from numpy import log, exp, power
from sklearn.base import BaseEstimator, ClassifierMixin
from data.import_data import tokenize, import_data
#from gensim.models.word2vec import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from difflib import get_close_matches

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


class ProxMarkovClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        """
        Called when initializing the classifier
        """

    def fit(self, sentences, labels):

        self.w2v = KeyedVectors.load_word2vec_format(
            './data/glove/glove.6B.50d.txt',
            binary=False)

        self.vocab = set()
        self.trans = {}
        self.logTable = {}

        distinct_labels = set(labels)

        self.labels = distinct_labels

        for l in distinct_labels:
            self.trans[l] = dict()

        for s, l in zip(sentences, labels):
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
        wn = get_close_matches(w, vocab, n=3)[0]

        self.trained_ = True
        self.falloff = 0.1
        return wn

    def compression(self, d):
        return exp(-self.falloff * power(d, 2))

    def nearest_word(self, w, words):
        if w in words:
            return (w, 1)
        else:
            nw = self.w2v.wv.most_similar_to_given(w, words)
            sim = self.w2v.wv.distance(w, nw)

            return (nw, self.compression(sim))

    def _sen_prob(self, s, l):
        self.runcount += 1
        if self.runcount % 25 == 0:
            print("RUNS:", self.runcount)

        tokens = tokenize(s, stop=False)
        pairings = zip(tokens, tokens[1:])

        trans = self.trans[l]

        prob = 0

        first_word = tokens[0]

        fwr, fwp = self.nearest_word(first_word, list(trans.keys()))

        wordprob = (sum(trans[fwr].values()) /
                    sum([sum(v.values()) for v in trans.values()]))
        prob += log(wordprob) + log(fwp)

        for w1, w2 in pairings:
            try:
                w1r, w1p = self.nearest_word(w1, list(trans.keys()))

                w2r, w2p = self.nearest_word(w2, list(trans[w1].keys()))
            except KeyError:
                w1 = self.fix_word(w1)
                w2 = self.fix_word(w2)

                w1r, w1p = self.nearest_word(w1, list(trans.keys()))

                w2r, w2p = self.nearest_word(w2, list(trans[w1].keys()))

            prob += log(trans[w1r][w2r] / sum(trans[w1r].values())) +\
                log(w1p) +\
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
        tr.text[:1000],
        tr.author[:1000],
        cv=3,
        n_jobs=-1)

    CM = confusion_matrix(
        tr.author,
        y_train_pred,
        labels=["EAP", "HPL", "MWS"])

    # Get prob dists across rows
    prob_CM = CM / CM.sum(axis=1, keepdims=True)

    print(prob_CM)
