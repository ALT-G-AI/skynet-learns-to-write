from data.import_data import import_data
from Predictors.windowed_G import windowedGClassifier
import pickle


if __name__ == '__main__':
    tr, te = import_data()

    myc = windowedGClassifier(encoder_size=30, DNNlayers=[30, 20], batch_n=3)
    myc.fit(tr.text, tr.author)

    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(myc, file)
