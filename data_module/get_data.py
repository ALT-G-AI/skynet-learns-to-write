from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import numpy as np

import cProfile

"""
if you just want to call a function which returns processed data then use get_data
this returns (train, test, crossval, onehot_object)

if you want a nice sklearn interface then use OneHot but it takes hours
"""


class OneHot(BaseEstimator, TransformerMixin):

    def __init__(self, sentence_length=50):
        self.sentence_length = sentence_length
        self.enc = LabelBinarizer(sparse_output=True)

    def clean_word(self, word):
        return ''.join([c for c in word if c.isalpha()])

    def reform_sentence(self, sen):
        """
        Reform a sentence string into a padded, terminated list of words
        """
        reformed_sen = [self.clean_word(w) for w in sen.lower().split(' ')]
        reformed_sen.append('.')
        if len(reformed_sen) > self.sentence_length:
            return None
        elif len(reformed_sen) < self.sentence_length:
            reformed_sen.extend(
                '\0' *
                (self.sentence_length - len(reformed_sen)))
        return reformed_sen

    def fit(self, X, y=None):
        # putting flat through a set so that the words are unique
        flat = set()
        for line in X:
            reformed = self.reform_sentence(line)
            if reformed is not None:
                for word in reformed:
                    flat.add(word)
        self.enc.fit(list(flat))

    def transform(self, X, y=None):
        out = list()
        for line in X:
            reformed = self.reform_sentence(line)
            if reformed is not None:
                # all the cpu time is spent in enc.transform
                out.append(self.enc.transform(reformed))
        return out

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        out = list()
        for line in X:
            out.append(
                [word for word in self.enc.inverse_transform(line) if word != ''])
        return out


def get_data(num_lines=None, infile='a.txt', outfile='one_hot_encoded.npy'):
    """
    num_lines only applies if generating a new outfile
    """
    try:
        # load previously encoded data
        (all_data, onehot) = np.load(outfile)
    except IOError:
        # encode the data ourselves
        onehot = OneHot()
        with open(infile, 'r', encoding='utf8') as inf:
            in_data = inf.readlines()[:num_lines]

        all_data = onehot.fit_transform(in_data)
        # save for later
        np.save(outfile, (all_data, onehot))

    # split into train/test/crossval
    crosval_prop = 0.2
    test_prop = 0.2
    intermediate_prop = crosval_prop + test_prop
    train_prop = 1 - intermediate_prop
    # always use the same random state so we are reproducible
    random_state = 42

    X_train, X_intermediate = train_test_split(
        all_data,
        test_size=intermediate_prop,
        random_state=random_state)

    X_crossval, X_test = train_test_split(
        X_intermediate,
        test_size=(crosval_prop / intermediate_prop),
        random_state=random_state)

    return (X_train, X_test, X_crossval, onehot)


def main():
    num_sentences = None

    (train, test, crossval, onehot) = get_data(num_lines=num_sentences)

    inverse = onehot.inverse_transform(train)

    print(inverse)

if __name__ == '__main__':
    # cProfile.run('main()')
    main()
