import numpy as np
from Processing.processing import tokenize


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
            if type(s) is str:
                yield tokenize(s), l
            else:
                yield s, l
        raise StopIteration('No more sentences to offer')

    def __call__(self):
        while(len(self.senbatchqueue) < self.batchsize):

            sen, lab = next(self.generator)

            def enc_with_uncommon(t):
                if t in self.encoder:
                    return self.encoder[t]
                else:
                    return self.encoder['#$UNCOMMON$#']

            encs = [enc_with_uncommon(t) for t in sen]

            for i in range(len(encs) + 1 - self.wlen):
                self.senbatchqueue.append(encs[i:i + self.wlen])
                self.labbatchqueue.append(lab)

        senbatch = self.senbatchqueue[0:self.batchsize]
        self.senbatchqueue = self.senbatchqueue[self.batchsize:]

        labbatch = self.labbatchqueue[0:self.batchsize]
        self.labbatchqueue = self.labbatchqueue[self.batchsize:]

        return {'windows': np.array(senbatch)}, [self.key[l] for l in labbatch]


a = None

if __name__ == '__main__':
    sens = ['a b a a b b c c a', 'a a a b c c a']
    labs = ['d', 'e']
    enc = {'a': 1, 'b': 2, 'c': 3}
    a = windowed_data(sens, labs, 3, 5, enc)
