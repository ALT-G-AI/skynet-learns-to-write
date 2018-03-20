from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

"""
if you just want to call a function which
returns processed data then use get_data

       this returns (train, test, crossval)
if you want a nice sklearn interface then use DataPipeline
"""


# numbers all words. Don't use this -
# it is just a prelimery so we can use OneHotEncoder

class WordNumberer(BaseEstimator, TransformerMixin):
    def __init__(self, sentence_length=50):
        self.sentence_length = sentence_length
        self.words = {'\0': 0}

    def reform_sentence(self, sen):
        """
        Reform a sentence string into a padded, terminated list of words
        """
        reformed_sen = sen.lower().split(' ')
        reformed_sen.append('.')
        if len(reformed_sen) > self.sentence_length:
                return None
        elif len(reformed_sen) < self.sentence_length:
                reformed_sen.extend(
                    '\0' *
                    (self.sentence_length - len(reformed_sen)))
        return reformed_sen

    def clean_data(self, d):
        """
        Reforms list of sentence strings into padded, terminated lists of words
        """
        return [self.reform_sentence(s) for s in d
                if self.reform_sentence(s) is not None]

    def _fit(self, data):
        # make dictionary of unique words, each with a number to represent them

        word_set = {w for line in data for w in line if w is not '\0'}

        self.words = {v: k for k, v in enumerate(word_set, 1)}
        self.words['\0'] = 0

        print("len(words) = %i" % len(self.words))

    def fit(self, X, y=None):
        data = self.clean_data(X)
        self._fit(data)

    def _transform(self, data):
        # replace with numbers
        proc_data = list()
        for line in data:
            proc_data.append([self.words[w] for w in line])

        return proc_data

    def transform(self, X):
        data = self.clean_data(X)
        return self._transform(data)

    def fit_transform(self, X, y=None):
        data = self.clean_data(X)
        print("len(data) = %i" % len(data))
        self._fit(data)
        return self._transform(data)


# pipeline from WordNumberer through OneHotEncoder.
class DataPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, sentence_length=50):
        # numberer
        self.numberer = WordNumberer(sentence_length)

        # one hot encoder
        self.enc = OneHotEncoder(dtype=int)

        # pipeline
        self.pipeline = Pipeline(steps=[
            ("word numberer", self.numberer),
            ("one-hot-encoding", self.enc),
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)

    def transform(self, X, y=None):
        self.pipeline.transform(X, y)

    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)


# This is probably what you want. Returns (train, test, crossval)
def get_data(file='a.txt'):
    pipe = DataPipeline()
    with open(file, 'r', encoding='utf8') as inf:
        all_data = pipe.fit_transform(inf.readlines())

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

    return (X_train, X_test, X_crossval)


# example
if __name__ == '__main__':
    (train, test, crossval) = get_data()
    for d in (train, test, crossval):
        print("shape = %s" % (d.shape,))
