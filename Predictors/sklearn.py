from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABC, abstractmethod

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


def test_sklearnclassifier(Clf, train_limit=None, **kwargs):
    from data.import_data import import_data
    from data.padded_sentences import PaddedSentenceTransformer
    from data.numbered_authors import NumberAuthorsTransformer
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

    tr, te = import_data()

    label_enc = NumberAuthorsTransformer()
    data_enc = PaddedSentenceTransformer(encoder_size=100)

    myc = Clf(**kwargs)

    if train_limit == None:
        train_limit = len(tr.text)

    X_train = data_enc.fit_transform(tr.text[:train_limit])
    y_train = label_enc.fit_transform(tr.author[:train_limit])

    print("Running cross-validation...")
    y_train_pred = cross_val_predict(myc, X_train, y_train)

    print("\nStatistics...")

    confusion = confusion_matrix(y_train, y_train_pred)
    print("Confucion Matrix:")
    print(confusion)
          # see textbook page 142. Ideally this should be non-zero only on the
          # diagonal (like an identity matrix)

    accuracy = accuracy_score(y_train, y_train_pred)
    print("Accuracy: {}".format(accuracy))

    precision = precision_score(y_train, y_train_pred, average='micro')
    print("Precision: {}".format(precision))

    recall = recall_score(y_train, y_train_pred, average='micro')
    print("Recall: {}".format(recall))

    f1 = f1_score(y_train, y_train_pred, average='micro')
    print("F1 Score: {}".format(f1))
