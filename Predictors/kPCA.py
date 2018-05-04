from sklearn.decomposition import KernelPCA
from sklearn.feature_extraction.text import CountVectorizer

from data.import_data import import_data
from data.LemmaCountVectorizer import LemmaCountVectorizer
from numpy import array

import matplotlib.pyplot as plt

train, ts = import_data()

# Storing the entire training text in a list
text = list(train.text.values)
authors = array(list(train.author.values))

tf_vectorizer = LemmaCountVectorizer(max_df=0.95,
                                min_df=2,
                                decode_error='ignore')

tf = tf_vectorizer.fit_transform(text)

#print(tf)

rbf_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.04)
x_reduced = rbf_pca.fit_transform(tf)

#print(x_reduced)

plt.figure()
plt.title("Original space")
reds = authors == "EAP"
blues = authors == "HPL"
greens = authors == "MWS"

plt.scatter(x_reduced[reds, 0], x_reduced[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(x_reduced[blues, 0], x_reduced[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.scatter(x_reduced[greens, 0], x_reduced[greens, 1], c="green",
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
