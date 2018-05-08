from Predictors.windowed_DNN import windowedDNN
from data.import_data import import_data
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score

import numpy as np


def logprint(*args):
    with open('win_DNN_grid.txt', 'a') as logfile:
        print(*args, file=logfile)


if __name__ == '__main__':
    tr, te = import_data()
    author_enum = {'HPL': 0, 'EAP': 1, 'MWS': 2}
    classed_auths = [author_enum[a] for a in tr.author]

    windows = [1, 2, 4, 6, 8, 10]
    layers = [
        [],
        [50],
        [100],
        [50, 25]]
    pte = [True,
           False]
    beta = [True,
            False]
    word_dim = [
        8,
        16,
        32,
        50]

    for wd in word_dim:
        logprint("Windows:", 5, "| Encoding:", wd, "| Layers:", [50])
        myc = windowedDNN(
            word_dim=wd,
            layers=[40],
            window=5,
            verbose=False,
            epochs=250,
            index_out=False)

        y_train_pred = cross_val_predict(
            myc,
            tr.text,
            classed_auths,
            cv=3,
            n_jobs=-1)

        logloss = log_loss(classed_auths, y_train_pred)

        logprint("Loss:", logloss)

        indexes = np.argmax(np.array(y_train_pred), 1)

        CM = confusion_matrix(
            classed_auths,
            indexes)

        prob_CM = CM / CM.sum(axis=1, keepdims=True)

        logprint(prob_CM)

        acc = accuracy_score(classed_auths, indexes)
        logprint("Acc:", acc)
        logprint("----------------------------------------\n")

    for w in windows:
        for p in pte:
            for l in layers:
                logprint(
                    "Windows:", w,
                    "| PTE:", p,
                    "| Layers:", l)

                myc = windowedDNN(
                    layers=l,
                    window=w,
                    pte=p,
                    verbose=False,
                    epochs=250,
                    index_out=False)

                y_train_pred = cross_val_predict(
                    myc,
                    tr.text,
                    classed_auths,
                    cv=3,
                    n_jobs=-1)

                logloss = log_loss(classed_auths, y_train_pred)

                logprint("Loss:", logloss)

                indexes = np.argmax(np.array(y_train_pred), 1)

                CM = confusion_matrix(
                    classed_auths,
                    indexes)

                prob_CM = CM / CM.sum(axis=1, keepdims=True)

                logprint(prob_CM)

                acc = accuracy_score(classed_auths, indexes)
                logprint("Acc:", acc)
                logprint("----------------------------------------\n")
