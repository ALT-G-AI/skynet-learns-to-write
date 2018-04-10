import numpy as np
from Processing.processing import tokenize


def predictor_data(sen, encoder, wlen):

    tokens = tokenize(sen)

    if len(tokens) < wlen:
        padding = [tokens[-1] for i in range(wlen - len(tokens))]
        tokens = tokens + padding

    def enc_with_uncommon(t):
        if t in encoder:
            return encoder[t]
        else:
            return encoder['#$UNCOMMON$#']

    windows = [
        [enc_with_uncommon(t) for t in tokens[i:i + wlen]]
        for i in range(len(tokens) + 1 - wlen)]

    return {'windows': np.array(windows)}


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
                tokens = tokenize(s)
                if len(tokens) < self.wlen:
                    padding = [
                        tokens[-1]
                        for i in range(self.wlen - len(tokens))]
                    tokens = tokens + padding
                yield tokens, l
            else:
                if len(s) < self.wlen:
                    padding = [
                        s[-1]
                        for i in range(self.wlen - len(s))]
                    s = s + padding
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
