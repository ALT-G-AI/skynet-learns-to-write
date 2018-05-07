from data.import_data import tokenize, import_data
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np


class PaddedSentenceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, sentence_length=50, encoder_size=100, padding_token='\0', unknown_token='$unknown$'):
        """
        sentence_length is in tokens
        """
        self.sentence_length = sentence_length
        self.encoder_size = encoder_size
        self.padding_token = padding_token
        self.unknown_token = unknown_token

    def pad_sentence_(self, sentence):
        """
        sentence should be tokenized
        """
        if len(sentence) > self.sentence_length:
            sentence = sentence[:self.sentence_length - 1]
            sentence.append('.')

        elif len(sentence) < self.sentence_length:
            for _ in range(self.sentence_length - len(sentence)):
                sentence.append(self.padding_token)

        # print(sentence)
        assert(len(sentence) == self.sentence_length)
        return sentence

    def encode_sentence_(self, tokenized_sentence):
        out = []
        for token in tokenized_sentence:
            try:
                out.append(self.encoder[token])
            except KeyError:
                out.append(self.encoder[self.unknown_token])

        if len(tokenized_sentence) != 50:
            print("")
            print(tokenized_sentence)
            print("")
            print(out)

        return out

    def process_data_(self, X):
        tokenized_sentences = [tokenize(s) for s in X]
        return [self.pad_sentence_(s) for s in tokenized_sentences]

    def fit_(self, padded_sentences):
        self.encoder = Word2Vec(
            # make sure the unknown token ends up in our vocabulary
            padded_sentences + [[self.unknown_token]],
            workers=8,
            min_count=0,
            max_vocab_size=None,
            size=self.encoder_size)

    def fit(self, X, y=None):
        padded_sentences = self.process_data_(X)

        self.fit_(padded_sentences)
        return self

    def transform_(self, padded_sentences):
        return np.array([self.encode_sentence_(s)
                         for s in padded_sentences])

    def fit_transform(self, X, y=None):
        padded_sentences = self.process_data_(X)
        self.fit_(padded_sentences)
        return self.transform_(padded_sentences)

    def transform(self, X, y=None):
        padded_sentences = self.process_data_(X)
        return self.transform_(padded_sentences)


if __name__ == '__main__':
    tr, te = import_data()

    count = len(tr.text)
    data = np.array(tr.text[:count])

    trans = PaddedSentenceTransformer()
    result = trans.fit_transform(data)

    for s, r in zip(data, result):
        assert(len(r) == trans.sentence_length)
