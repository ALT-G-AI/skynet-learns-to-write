import matplotlib.pyplot as plt
# This import is used even though PyCharm thinks it is not
# In case deleted it is this import "from mpl_toolkits.mplot3d import Axes3D"
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
from sklearn.decomposition import KernelPCA
from sklearn.feature_extraction.text import CountVectorizer

from data.import_data import import_data

train, ts = import_data()

# Storing the entire training text in a list
text = list(train.text.values)
authors = array(list(train.author.values))

tf_vectorizer = CountVectorizer(max_df=0.95,
                                min_df=2,
                                # stop_words='english',
                                decode_error='ignore')

tf = tf_vectorizer.fit_transform(text)

# print(tf)

rbf_pca = KernelPCA(n_components=3, kernel="sigmoid", gamma=0.04)
x_reduced = rbf_pca.fit_transform(tf)

# print(x_reduced)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title("Original space")
reds = authors == "EAP"
blues = authors == "HPL"
greens = authors == "MWS"

ax.scatter(x_reduced[reds, 0], x_reduced[reds, 1], x_reduced[reds, 2], c="red",
           s=20, edgecolor='k')
ax.scatter(x_reduced[blues, 0], x_reduced[blues, 1], x_reduced[blues, 2], c="blue",
           s=20, edgecolor='k')
ax.scatter(x_reduced[greens, 0], x_reduced[greens, 1], x_reduced[greens, 2], c="green",
           s=20, edgecolor='k')

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.show()
