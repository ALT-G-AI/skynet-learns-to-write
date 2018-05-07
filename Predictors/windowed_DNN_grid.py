from Predictors.windowed_DNN import windowedDNN
from data.import_data import import_data
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


def logprint(*args):
    with open('win_DNN_grid.txt', 'a') as logfile:
        print(*args, file=logfile)


if __name__ == '__main__':
    tr, te = import_data()
    author_enum = {'HPL': 0, 'EAP': 1, 'MWS': 2}
    classed_auths = [author_enum[a] for a in tr.author]

    windows = [1, 4, 7]
    layers = [
        [25],
        [50],
        [50, 20]]
    word_dim = [
        8,
        16,
        32,
        50]

    for w in windows:
        for wd in word_dim:
            for l in layers:
                logprint("Windows:", w, "| Encoding:", wd, "| Layers:", l)

                myc = windowedDNN(
                    word_dim=wd,
                    layers=l,
                    window=w,
                    verbose=True,
                    epochs=100)

                y_train_pred = cross_val_predict(
                    myc,
                    tr.text,
                    classed_auths,
                    cv=3,
                    n_jobs=-1)

                CM = confusion_matrix(
                    classed_auths,
                    y_train_pred)

                prob_CM = CM / CM.sum(axis=1, keepdims=True)

                logprint(prob_CM)
                logprint("----------------------------------------\n")
