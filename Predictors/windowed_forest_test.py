from data.import_data import import_data
from Predictors.windowed_forest import windowedForestClassifier

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix

import pickle
import numpy as np

filename = 'trained_model_forest.pkl'

# from
# https://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
def most_common(lst):
    return max(set(lst), key=lst.count)

if __name__ == '__main__':
    # we need testing data either way so import now
    tr, te = import_data()

    try:
        # try to load data
        with open(filename, 'rb') as file:
            myc = pickle.load(file)

    except IOError:
        print("Re-training")

        myc = windowedForestClassifier(encoder_size=30)

        myc.fit(tr.text, tr.author)

        with open(filename, 'wb') as file:
            pickle.dump(myc, file)

    # one way or another we now have a trained model
    # test the trained model on the testing data

    # train_pred = cross_val_predict(myc, te['text'], te['author'], cv=3)
# train_acc = cross_val_score(myc, np.array(te['text']),
# np.array(te['author']), cv=3, scoring="accuracy")

    X_test = np.array(te['text'])
    y_test = np.array(te['author'])

    count_exep = 0
    num = 1000
    incorrect = 0
    for i in range(num):
        if (i % 100) == 0:
            print("\niter: {}/{}\n".format(i, num))
        try:
            pred = most_common(myc.predict(X_test[i]))
            correct = y_test[i]
            if pred != correct:
                incorrect += 1

            print("Predicted {}, label was {}".format(pred, correct))
        except:
            count_exep += 1

    print("{} exception sentences out of {}".format(count_exep, num))
    print("Accuracy is {}".format(1.0 - float(incorrect)/float((num - count_exep))))

