from data.import_data import import_data
from Predictors.windowed_G import windowedGClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    tr, te = import_data()

    myc = windowedGClassifier(
        encoder_size=50,
        DNNlayers=[50, 25],
        batch_n=500,
        training_steps=200)

    y_train_pred = cross_val_predict(myc, tr['text'], tr['author'], cv=2)

    CM = confusion_matrix(
        tr['author'],
        y_train_pred,
        labels=["EAP", "HPL", "MWS"])

    prob_CM = CM / CM.sum(axis=1, keepdims=True)

    print(prob_CM)

    # myc.fit(tr['text'], tr['author'])

    # with open('trained_model.pkl', 'wb') as file:
    #     pickle.dump(myc, file)
