from data.import_data import import_data
from Predictors.windowed_G import windowedGClassifier
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
import pickle

def tokenize(sen):
    return word_tokenize(sen.lower())

ext_encoded = ""
ext_authors = ""

batch_n = 3

if __name__ == '__main__':
    tr, te = import_data()

    text = tr['text'].as_matrix()
    authors = tr['author'].as_matrix()

    author_set = set(authors)
    author_key = {k: v for k,v in zip(author_set, range(len(author_set)))}

    authors = np.array([author_key[a] for a in authors])

    encsize = 2
    window = 5

    tokens = [tokenize(s) for s in text]

    encoder = Word2Vec(tokens,
        size=encsize,
        window=window,
        min_count=0,
        workers=4)

    encoded = np.array([np.array([encoder.wv[w] for w in s]) for s in tokens])

    ext_encoded = []
    ext_authors = []

    for s, a in zip(encoded, authors):
        for i in range(len(s) + 1 - window):
            ext_encoded.append(s[i:i + window])
            ext_authors.append(a)

    ext_encoded = np.array(ext_encoded)
    ext_authors = np.array(ext_authors)

    #print(encoded[0])

    dataset_full = tf.data.Dataset.from_tensor_slices((ext_encoded, ext_authors))
    dataset_batched = dataset_full.batch(batch_n)
    iterator = dataset_batched.make_one_shot_iterator()
    el = iterator.get_next()

    with tf.Session() as sess:
         print(sess.run(el))

    # myc = windowedGClassifier(encoder_size=30, DNNlayers=[30, 20], batch_n = 3)
    # myc.fit(tr.text[0:2], tr.author[0:1])

    # with open('trained_model.pkl', 'wb') as file:
    #     pickle.dump(myc, file)
