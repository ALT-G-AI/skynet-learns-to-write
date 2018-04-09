from data.import_data import import_data
from Predictors.windowed_forest import windowedForestClassifier
import pickle


if __name__ == '__main__':
    tr, te = import_data()

    myc = windowedForestClassifier(encoder_size=30)

    myc.fit(tr.text, tr.author)

    with open('trained_model_forest.pkl', 'wb') as file:
        pickle.dump(myc, file)
