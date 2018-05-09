from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from Predictors.probabilistic import ProbabilisticClassifier
from data.import_data import import_data

if __name__ == '__main__':
    tr, te = import_data()

    myc = ProbabilisticClassifier(beta_method=False, stem=True, lemma=True)
    y_train_pred = cross_val_predict(myc, tr.text, tr.author, cv=3)

    CM = confusion_matrix(
        tr.author,
        y_train_pred,
        labels=["EAP", "HPL", "MWS"])

    # Get prob dists across rows
    prob_CM = CM / CM.sum(axis=1, keepdims=True)

    print(prob_CM)
