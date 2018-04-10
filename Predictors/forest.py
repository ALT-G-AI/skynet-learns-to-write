from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

import numpy as np

# doens't really do much. See data.{padded_sentences, named_authors} for
# data processing


class ForestClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.forest_clf = RandomForestClassifier(
            n_estimators=500,
            max_leaf_nodes=16,
            n_jobs=-1)

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

        print("Training Forest")
        self.forest_clf.fit(sentences, labels)

        return self

    def predict(self, X):
        return self.forest_clf.predict(self.reshape_(X))

filename = 'trained_model_forest.pkl'

if __name__ == '__main__':
    from data.import_data import import_data
    from data.padded_sentences import PaddedSentenceTransformer
    from data.numbered_authors import NumberAuthorsTransformer

    tr, te = import_data()

    label_enc = NumberAuthorsTransformer()
    data_enc = PaddedSentenceTransformer(encoder_size=100)

    myc = ForestClassifier()

    data_train = data_enc.fit_transform(tr.text)
    labels_train = label_enc.fit_transform(tr.author)

    myc.fit(data_train, labels_train)

    # test the trained model on the testing data
    # very rudimentary test for now

    X_test = np.array(te.text)
    y_test = np.array(te.author)

    count_exep = 0
    num = 5000
    incorrect = 0
    for i in range(num): 
        if (i % 100) == 0:
            print("\niter: {}/{}\n".format(i, num))

        try:
            enc = data_enc.transform([X_test[i]])
            pred = label_enc.inverse_transform(myc.predict(enc))

            correct = y_test[i]
            if pred[0] != correct:
                incorrect += 1
            
            print("Predicted {}, label was {}\n".format(pred, correct))

        except Exception as e:
            # catch encoding exceptions from unknown words (TODO)
            #print(e)
            count_exep += 1

        

    print("{} exception sentences out of {}".format(count_exep, num))
    print("Accuracy is {}".format(1.0 - float(incorrect)/float((num - count_exep))))

