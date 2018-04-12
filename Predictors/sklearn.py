from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABC, abstractmethod

from data.import_data import import_data
from data.padded_sentences import PaddedSentenceTransformer
from data.numbered_authors import NumberAuthorsTransformer
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

import numpy as np

# doens't really do much. See data.{padded_sentences, named_authors} for
# data processing


class SklearnClassifier(ABC, BaseEstimator, ClassifierMixin):

    """
    generic wrapper for sklearn classifiers
    """

    @abstractmethod
    def init_clf_(self, **kwargs):
        """
        creates self.clf
        """
        pass

    def __init__(self, **kwargs):
        self.init_clf_(**kwargs)

    @staticmethod
    def reshape_(arr):
        # input data is (num_sentences, sentence_length, word_encoding_length)
        # forest_clf only does 2D data so reshape
        arr = np.array(arr)
        return np.reshape(arr, (arr.shape[0], arr.shape[1] * arr.shape[2]))

    def fit(self, sentences, labels):
        """
        sentences and labels should already be encoded
        """
        sentences = self.reshape_(sentences)

        print("Training Classifier")
        self.clf.fit(sentences, labels)

        return self

    def predict(self, X):
        return self.clf.predict(self.reshape_(X))

    def get_params(self, deep=True):
        return self.clf.get_params(deep)


def get_data_(train_limit, AuthorProc):
    tr, te = import_data()

    # cross_val_predict (test_sklearnclassifier) does the train/test/cv split
    # internally
    text = list(tr.text) + list(te.text)
    author = list(tr.author) + list(te.author)

    label_enc = AuthorProc()
    data_enc = PaddedSentenceTransformer(encoder_size=100)

    if train_limit == None:
        train_limit = len(text)

    X = data_enc.fit_transform(text[:train_limit])
    y = label_enc.fit_transform(author[:train_limit])

    return (X, y)


def test_estimator_(myc, AuthorProc, train_limit):
    X_train, y_train = get_data_(train_limit, AuthorProc)

    print("Running cross-validation...")
    y_train_pred = cross_val_predict(myc, X_train, y_train)

    print("\nStatistics...")

    try:
        confusion = confusion_matrix(y_train, y_train_pred)
        print("Confucion Matrix:")
        print(confusion)
        # see textbook page 142. Ideally this should be non-zero only on the
        # diagonal (like an identity matrix)
    except ValueError:
        print("Confusion matrix not supported for that encoding")
        # Probably to do with one-hot ecoding
        # ValueError: multilabel-indicator is not supported
        # https://stackoverflow.com/questions/42950705/valueerror-cant-handle-mix-of-multilabel-indicator-and-binary

    accuracy = accuracy_score(y_train, y_train_pred)
    print("Accuracy: {}".format(accuracy))

    precision = precision_score(y_train, y_train_pred, average='micro')
    print("Precision: {}".format(precision))

    recall = recall_score(y_train, y_train_pred, average='micro')
    print("Recall: {}".format(recall))

    f1 = f1_score(y_train, y_train_pred, average='micro')
    print("F1 Score: {}".format(f1))


def test_sklearnclassifier(Clf, AuthorProc=NumberAuthorsTransformer, train_limit=None, **kwargs):
    myc = Clf(**kwargs)

    test_estimator_(myc, AuthorProc, train_limit)


def random_search_params(Clf, param_dist, AuthorProc=NumberAuthorsTransformer, train_limit=None, n_iter=100):
    X, y = get_data_(train_limit, AuthorProc)
    my_clf = Clf()

    search = RandomizedSearchCV(my_clf, param_dist, n_iter=n_iter, n_jobs=-1)

    print("Running search...")
    search.fit(X, y)

    print("\nBest Parameters: {}".format(search.best_params_))

    print("\nTesting best estimator...")
    test_estimator_(search.best_estimator_, AuthorProc, train_limit)
