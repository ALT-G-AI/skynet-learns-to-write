from Predictors.probabilistic import ProbabilisticClassifier
from data.import_data import import_data
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from numpy import argmax
if __name__ == '__main__':
    tr, te = import_data()

    author_trans = {"EAP": 0, "HPL": 1, "MWS": 2}
    back_auth = {v: k for k, v in author_trans.items()}

    myc = ProbabilisticClassifier(
        beta_method=True,
        stem=True,
        lemma=True,
        index_out=False,
        labels=["EAP", "HPL", "MWS"])

    class_auths = [author_trans[a] for a in tr.author]

    y_train_pred = cross_val_predict(myc, tr.text, tr.author, cv=3)

    names_out = [back_auth[a] for a in argmax(y_train_pred, 1)]

    CM = confusion_matrix(
        tr.author,
        names_out,
        labels=["EAP", "HPL", "MWS"])

    # Get prob dists across rows
    prob_CM = CM / CM.sum(axis=1, keepdims=True)

    print(CM)
    print("Acc:", accuracy_score(tr.author, names_out))
    print("Loss:", log_loss(class_auths, y_train_pred))
    print(prob_CM)
