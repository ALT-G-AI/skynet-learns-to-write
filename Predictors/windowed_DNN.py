import numpy as np
from gensim.models import Word2Vec
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (confusion_matrix,
                             log_loss,
                             accuracy_score)
from sklearn.model_selection import cross_val_predict

from data.glove.pre_encoder import pte
from data.import_data import import_data
from data.pipelines import (tokenize_pipe,
                            lower_pipe,
                            stem_pipe,
                            lemmatize_pipe,
                            uncommon_pipe,
                            encode_pipe,
                            window_pipe,
                            window_pipe_nolabel,
                            cull_words_pipe)


class WindowedDNN(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            window=5,
            layers=[50, 25],
            word_dim=50,
            epochs=250,
            batch=100,
            verbose=True,
            pte=False,
            index_out=True):
        """
        Called when initializing the classifier
        """
        self.window = window
        self.layers = layers
        self.word_dim = word_dim
        self.epochs = epochs
        self.batch = batch
        self.verbose = verbose
        self.pte = pte
        self.index_out = index_out

    def fit(self, sentences, labels):

        if self.verbose:
            print("Building NN")
        model = Sequential()
        model.add(Flatten(input_shape=(self.window, self.word_dim)))

        for l in self.layers:
            model.add(Dense(l, activation='relu'))

        model.add(Dense(3, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        self.model = model

        if self.verbose:
            model.summary()

        p1 = lower_pipe(sentences)
        p2 = tokenize_pipe(p1)
        p3 = stem_pipe(p2)
        p4 = lemmatize_pipe(p3)
        p5 = uncommon_pipe(p4)

        clean_sens = list(p5)
        self.clean_sens = clean_sens

        if self.verbose:
            print("Building word embedding")

        self.encoder = None
        if self.pte:
            self.encoder = pte()
        else:
            self.encoder = Word2Vec(
                clean_sens,
                size=self.word_dim,
                min_count=0)

        enc = encode_pipe(clean_sens, self.encoder)
        self.enc = enc
        windows = list(window_pipe(enc, labels, self.window))

        win_sens = [w[0] for w in windows]
        win_labs = [w[1] for w in windows]

        y_inp = to_categorical(win_labs)

        if self.verbose:
            print("Training NN")

        model.fit(
            np.array(win_sens),
            np.array(y_inp),
            epochs=self.epochs,
            batch_size=self.batch,
            verbose=self.verbose)

    def _pred_sen(self, s):
        s_array = [s]
        p1 = lower_pipe(s_array)
        p2 = tokenize_pipe(p1)
        p3 = stem_pipe(p2)
        p4 = lemmatize_pipe(p3)
        p5 = p4
        if not self.pte:
            p5 = cull_words_pipe(p4, self.encoder.wv.vocab)
        p6 = encode_pipe(p5, self.encoder)
        windows = np.array(list(window_pipe_nolabel(p6, self.window)))

        preds = self.model.predict(windows, batch_size=len(windows))

        logs = np.log(preds)
        flat = np.sum(logs, 0)

        winner_index = np.argmax(flat)
        if self.index_out:
            return winner_index
        else:
            return flat

    def predict(self, X):
        return [self._pred_sen(s) for s in X]


if __name__ == '__main__':
    tr, te = import_data()
    author_enum = {'HPL': 0, 'EAP': 1, 'MWS': 2}

    classed_auths = [author_enum[a] for a in tr.author]

    myc = WindowedDNN(
        epochs=250,
        layers=[100],
        window=5,
        pte=True,
        index_out=False)

    y_train_pred = cross_val_predict(
        myc,
        tr.text,
        classed_auths,
        cv=3,
        n_jobs=-1)

    indices = np.argmax(np.array(y_train_pred), 1)

    print("Loss:", log_loss(classed_auths, y_train_pred))
    print("Acc:", accuracy_score(classed_auths, indices))

    CM = confusion_matrix(
        classed_auths,
        indices)

    # Get prob dists across rows
    prob_CM = CM / CM.sum(axis=1, keepdims=True)

    print(prob_CM)
