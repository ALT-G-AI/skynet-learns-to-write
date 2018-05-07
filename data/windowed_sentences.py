import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

from data.import_data import tokenize, import_data


class WindowedSentenceTransformer(BaseEstimator, TransformerMixin):
    """
    Sentence Windowing. Discards leftover tokens (no padding)
    
    Labels are compulsory. Returns
    [(window1, author), (window2, author), (windowN, author), (sentence2window1, author2), ...]
    """

    def __init__(self, window_length=5, encoder_size=100, unknown_token='$unknown$', encoder_class = Word2Vec):
        """
        window_length is in tokens
        """
        self.window_length = window_length
        self.encoder_size = encoder_size
        self.unknown_token = unknown_token
        self.encoder_class = encoder_class

    def encode_sentence_(self, tokenized_sentence):
        out = []
        for token in tokenized_sentence:
            try:
                out.append(self.encoder[token])
            except KeyError:
                out.append(self.encoder[self.unknown_token])

        return out

    def window_sentence_(self, sentence):
        return list(zip(*[sentence[i::self.window_length] for i in range(self.window_length)]))

    def process_data_(self, X, y):
        tokenized_sentences = [tokenize(s) for s in X]
        data = [self.window_sentence_(s) for s in tokenized_sentences]
        return [(sen, author) for lst, author in zip(data, y) for sen in lst]

    def fit_(self, windowed_sentences):
        if self.encoder_class is Word2Vec:
            self.encoder = Word2Vec(
                # make sure the unknown token ends up in our vocabulary
                [b for a in windowed_sentences for b in a] + [[self.unknown_token]],
                workers=8,
                min_count=0,
                max_vocab_size=None,
                size=self.encoder_size)
        else:
            self.encoder = self.encoder_class()

    def fit(self, X, y):
        windowed_sentences = self.process_data_(X, y)
        self.fit_(windowed_sentences)
        return self

    def transform_(self, padded_sentences):
        return np.array([self.encode_sentence_(s)
                         for s in padded_sentences])

    def fit_transform(self, X, y):
        windowed_sentences = self.process_data_(X, y)
        # return windowed_sentences # for debugging only
        windows, labels = zip(*windowed_sentences)  # zip is the inverse of itself
        self.fit_(windows)
        enc_windows = self.transform_(windows)
        return zip(enc_windows, labels)

    def transform(self, X, y):
        windowed_sentences = self.process_data_(X, y)
        windows, labels = zip(*windowed_sentences)
        enc_windows = self.transform_(windows)
        return zip(enc_windows, labels)


if __name__ == '__main__':
    tr, te = import_data()

    count = 100
    data = np.array(tr.text[:count])
    labels = np.array(tr.author[:count])

    trans = WindowedSentenceTransformer()
    result = trans.fit_transform(data, labels)

    # need to turn off encoding to make this look sensible
    # to do this just return windowed_sentences from fit_transform
    for s in result:
        print(s)
        print("")
