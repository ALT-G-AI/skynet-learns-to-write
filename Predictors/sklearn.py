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

def test_sklearnclassifier(Clf, **kwargs):
    from data.import_data import import_data
    from data.padded_sentences import PaddedSentenceTransformer
    from data.numbered_authors import NumberAuthorsTransformer

    tr, te = import_data()

    label_enc = NumberAuthorsTransformer()
    data_enc = PaddedSentenceTransformer(encoder_size=100)

    myc = Clf(**kwargs)

    data_train = data_enc.fit_transform(tr.text)
    labels_train = label_enc.fit_transform(tr.author)

    myc.fit(data_train, labels_train)

    # test the trained model on the testing data
    # very rudimentary test for now

    X_test = np.array(te.text)
    y_test = np.array(te.author)

    num = 1000 #len(te.text)
    incorrect = 0
    for i in range(num): 
        if (i % 100) == 0:
            print("\niter: {}/{}\n".format(i, num))

        enc = data_enc.transform([X_test[i]])
        pred = label_enc.inverse_transform(myc.predict(enc))

        correct = y_test[i]
        if pred[0] != correct:
            incorrect += 1

        print("Predicted {}, label was {}".format(pred, correct))

    print("Accuracy is {}".format(1.0 - float(incorrect)/float((num - count_exep))))

