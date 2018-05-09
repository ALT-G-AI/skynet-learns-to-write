import numpy as np
from gensim.models import KeyedVectors


class pte():
    def __init__(self):
        """
        Called when initializing the classifier
        """
        print("Loading W2V pre-trained model")
        self.w2v = KeyedVectors.load_word2vec_format(
            './data/glove/glove.6B.50d.txt',
            binary=False)

        print("Calculating mean vector")
        self.mean = np.mean(
            [self.w2v.wv[w] for w in self.w2v.wv.vocab.keys()],
            0)

    def __getitem__(self, n):
        try:
            out = self.w2v[n]
        except KeyError:
            out = self.mean
        return out
