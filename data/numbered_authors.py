from data.import_data import import_data
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

class NumberAuthorsTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        authors = frozenset(X)
        self.key = {k: v for k, v in zip(authors, range(len(authors)))}
        self.inv_key = {k: v for v, k in zip(authors, range(len(authors)))}
        return self

    def transform(self, X, y=None):
        return [self.key[author] for author in X]

    def inverse_transform(self, X, y=None):
        return [self.inv_key[enc_author] for enc_author in X]

if __name__ == '__main__':
    tr, te = import_data()

    authors = np.array(tr.author)

    trans = NumberAuthorsTransformer()

    result = trans.fit_transform(authors)
    inverted = trans.inverse_transform(result)

    for a, b in zip(authors, inverted):
        assert(a == b)


