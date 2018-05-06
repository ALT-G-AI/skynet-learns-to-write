from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from data.import_data import import_data


def tokenize_pipe(prior):
    for s in prior:
        yield word_tokenize(s)


def lower_pipe(prior):
    for s in prior:
        yield s.lower()


def strip_stopwords_pipe(prior):
    for s in prior:
        yield [
            w for w in s
            if w not in stopwords.words('english')]


def stem_pipe(prior):
    if not hasattr(stem_pipe, 'ps'):
        stem_pipe.ps = PorterStemmer()
    for s in prior:
        yield [stem_pipe.ps.stem(w) for w in s]


def lemmatize_pipe(prior):
    if not hasattr(lemmatize_pipe, 'wnl'):
        lemmatize_pipe.wnl = WordNetLemmatizer()

    for s in prior:
        yield [lemmatize_pipe.wnl.lemmatize(w) for w in s]


def uncommon_pipe(prior, thresh=1):
    full = list(prior)
    count = Counter([w for s in full for w in s])

    def repl(w):
        if count[w] <= thresh:
            return '!#RARE'
        else:
            return w

    for s in full:
        yield [repl(w) for w in s]


if __name__ == '__main__':
    tr, te = import_data()

    sens = tr.text

    p1 = lower_pipe(sens)
    p2 = tokenize_pipe(p1)
    p3 = stem_pipe(p2)
    p4 = lemmatize_pipe(p3)
    p5 = uncommon_pipe(p4, 1)
    for i in p5:
        print(i)
