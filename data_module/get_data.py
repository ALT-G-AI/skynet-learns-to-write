from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# if you just want to call a function which returns processed data then use get_data
# if you want a nice sklearn interface then use DataPipeline

# numbers all words. Don't use this - it is just a prelimery so we can use OneHotEncoder
class WordNumberer(BaseEstimator, TransformerMixin):
    def __init__(self, sentence_length=50):
        self.sentence_length = sentence_length
        self.words = {'\0': 0}

    def clean_word(self, w):
        return ''.join([ c.lower() for c in w if c.isalpha() ])

    def clean_data(self, d):
        data = list()
        for line in d:
            clean_line = [self.clean_word(w) for w in line.split(' ')]
            clean_line.append('.') # give every sentence a termination symbol

            # sentences need to have a fixed length for most of sklearn's stuff
            if len(clean_line) > self.sentence_length:
                continue
            elif len(clean_line) < self.sentence_length:
                clean_line.extend('\0'*(self.sentence_length - len(clean_line)))

            data.append(clean_line)

        return data

    def _fit(self, data):
        # make dictionary of unique words, each with a number to represent them
        index = 1 # 0 is always the line ending character
        for line in data:
            for word in line:
                if (word in self.words):
                    continue
                else:
                    self.words[word] = index
                    index += 1


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

# This is probably what you want
def get_data(file='data_module/a.txt'):
    pipe = DataPipeline()
    with open(file, 'r', encoding='utf8') as inf:
        return pipe.fit_transform(inf.readlines())
        
# example
if __name__ == '__main__':
    sparse = get_data()
    print(sparse[0])
