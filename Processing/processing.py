from nltk.tokenize import word_tokenize
from collections import Counter


def tokenize(sen):
    return word_tokenize(sen.lower())


def merge_uncommon_words(sens, thresh):
    print("Tokenizing")
    tokenized_sens = [tokenize(s) for s in sens]
    print("Counting")
    counted = Counter([w for s in tokenized_sens for w in s])

    def fix(w):
        if counted[w] <= thresh:
            return '#$UNCOMMON$#'
        else:
            return w
    print("Evaluating Uncommon Words")
    return [[fix(w) for w in s] for s in tokenized_sens]
