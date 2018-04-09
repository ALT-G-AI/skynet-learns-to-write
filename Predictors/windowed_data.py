from nltk.tokenize import word_tokenize
import numpy as np


class windowed_data():
    def __init__(self, sentences, labels, wlen, batchsize, encoder):
        self.sentences = sentences
        self.labels = labels
        self.wlen = wlen
        self.batchsize = batchsize
        self.encoder = encoder

        authors = set(labels)
        self.key = {k: v for k, v in zip(authors, range(len(authors)))}

        self.senbatchqueue = []
        self.labbatchqueue = []
        self.generator = self.get_next_sentence_gen_()

    def get_next_sentence_gen_(self):
        for s, l in zip(self.sentences, self.labels):
            yield self.tokenize_(s), l
        raise StopIteration('No more sentences to offer')

    @staticmethod
    def tokenize_(sen):
        return word_tokenize(sen.lower())

    def __call__(self):
        while(len(self.senbatchqueue) < self.batchsize):

            sen, lab = next(self.generator)

            encs = [self.encoder[t] for t in sen]

            for i in range(len(encs) + 1 - self.wlen):
                self.senbatchqueue.append(encs[i:i + self.wlen])
                self.labbatchqueue.append(lab)

        senbatch = self.senbatchqueue[0:5]
        self.senbatchqueue = self.senbatchqueue[5:]

        labbatch = self.labbatchqueue[0:5]
        self.labbatchqueue = self.labbatchqueue[5:]

        return {'windows': np.array(senbatch)}, [self.key[l] for l in labbatch]


a = None

if __name__ == '__main__':
    sens = ['a b a a b b c c a', 'a a a b c c a']
    labs = ['d', 'e']
    enc = {'a': 1, 'b': 2, 'c': 3}
    a = windowed_data(sens, labs, 3, 5, enc)
