from data.import_data import import_data
from Predictors.windowed_G import windowedGClassifier
from nltk.tokenize import word_tokenize
import pickle

def tokenize(sen):
    return word_tokenize(sen.lower())

if __name__ == '__main__':
    tr, te = import_data()

    
    
    encoder = Word2Vec([
        tokenize(s)
        for s in sentences],
        size=self.encoder_size,
        window=self.window,
        min_count=0,
        workers=4)

    myc = windowedGClassifier(encoder_size=30, DNNlayers=[30, 20], batch_n = 3)
    myc.fit(tr.text[0:2], tr.author[0:1])

    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(myc, file)
