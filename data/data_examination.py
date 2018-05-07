from data.import_data import import_data
from data.pipelines import (tokenize_pipe,
                            lower_pipe,
                            stem_pipe,
                            lemmatize_pipe,
                            uncommon_pipe,
                            encode_pipe,
                            window_pipe,
                            window_pipe_nolabel,
                            cull_words_pipe,
                            strip_stopwords_pipe)
from collections import Counter

import matplotlib.pyplot as plt


tr, te = import_data()

sens = list(tr['text'])
authors = list(tr['author'])

n_auths = Counter(authors)

s_by_a = {a: [s for s, a1 in zip(sens, authors) if a1 == a] for a in n_auths}

tok_s_by_a = {
    k:
    list(tokenize_pipe(lower_pipe(v))) for k, v in s_by_a.items()}


def senlens(X):
    out = Counter([len(s) for s in X])
    return out


def wordcounts(X):
    ctr = Counter([w for s in X for w in s])
    return ctr


def wordfreq(X):
    count = wordcounts(X)
    sumc = sum(count.values())
    return {k: v / sumc for k, v in count.items()}


def wordbeta(X):
    count = wordcounts(X)
    sumc = sum(count.values())

    def beta(o):
        cnt = count[o]
        alpha = 1 + cnt
        beta = 1 + sumc - cnt

        return alpha / (alpha + beta)

    return {k: beta(k) for k in count.keys()}


if __name__ == '__main__':
    f1 = plt.figure(1)
    for a, d in tok_s_by_a.items():
        counts = senlens(d)
        sumcounts = sum(counts.values())
        base = range(1, max(counts.keys()) + 1)
        plt.semilogx(base, [counts[x] / sumcounts for x in base], label=a)
    plt.legend()
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')

    f2 = plt.figure(2)
    for i, (a, d) in enumerate(tok_s_by_a.items()):
        plt.subplot(3, 1, i + 1)
        counts = wordcounts(d)
        top = counts.most_common(15)
        wds = [k for k, v in top]
        vs = [v for k, v in top]
        vs = [x / sum(vs) for x in vs]
        plt.bar(wds, vs)
        plt.title(a)
        plt.subplots_adjust(hspace=1)
        plt.xticks(rotation=45)

    f2 = plt.figure(3)
    for i, (a, d) in enumerate(tok_s_by_a.items()):
        print('Stopwords removal:', a)
        culled = list(strip_stopwords_pipe(d))
        plt.subplot(3, 1, i + 1)
        counts = wordcounts(culled)
        top = counts.most_common(15)
        wds = [k for k, v in top]
        vs = [v for k, v in top]
        vs = [x / sum(vs) for x in vs]
        plt.bar(wds, vs)
        plt.title(a)
        plt.subplots_adjust(hspace=1)
        plt.xticks(rotation=45)

    plt.show()