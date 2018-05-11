import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from Predictors.windowed_DNN import WindowedDNN
from Predictors.probabilistic_NN import ProbabilisticNNClassifier
from math import floor

class DNNEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble combining windowed_DNN and probabilistic_NN using a single dense layer
    """

    def __init__(self, windowed_DNN_args, probabilistic_NN_args, train_split = 0.9):
        """
        train_split = proportion of training data used to train component networks
        *_args = dictionary of kwargs for that model
        """
        self.train_split = 0.9

        self.windowed_DNN = WindowedDNN(**windowed_DNN_args)
        self.probabilistic_NN = ProbabilisticNNClassifier(**probabilistic_NN_args)

        self.sess = tf.Session()

        self.X = tf.placeholder(tf.float32, [None, 6]) # 2 classifiers each outputting 3 probabilities
        self.y = tf.placeholder(tf.int64, [None])

        out_layer = tf.layers.dense(self.X, 3, activation=tf.nn.sigmoid)

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=out_layer)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        self.probs = tf.nn.softmax(out_layer)
        _, self.output = tf.nn.top_k(out_layer, k=1)

        self.training_op = optimizer.minimize(loss)
        self.correct = tf.nn.in_top_k(out_layer, self.y, 1)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        self.init = tf.global_variables_initializer()

    def __del__(self):
        try:
            self.sess.close()
        except AttributeError:
            pass

    def pred_(self, sentences):
        dnn = self.windowed_DNN.predict(sentences)
        prob = self.probabilistic_NN.predict(sentences)
        return [np.append(i,j) for i,j in zip(dnn, prob)]

    def fit(self, sentences, labels):
        final_layer_prop = 1 - self.train_split

        split = floor(final_layer_prop * sentences.shape[0])
        X_train = np.array(sentences[split:])
        y_train = np.array(labels[split:])

        print("\n\nTraining probabilistic classifier")
        self.probabilistic_NN.fit(sentences[:split], labels[:split])
        print("\n\nTraining windowed DNN")
        self.windowed_DNN.fit(sentences[:split], labels[:split])

        print("\n\nTraining final layer")

        self.sess.run(self.init)
        for epoch in range(500):
            try:
                for iteration in range(X_train.shape[0] // 200): 
                    batch_choices = np.random.choice(X_train.shape[0], 200)
                    X_batch = X_train[batch_choices]
                    y_batch = y_train[batch_choices]

                    self.sess.run(self.training_op, feed_dict={self.X: self.pred_(X_batch), self.y: y_batch})

                acc_train = self.sess.run(self.accuracy, feed_dict={self.X: self.pred_(X_batch), self.y: y_batch})
                print("{}/{}\t\tTrain accuracy:\t{:.3f}".format(epoch+1, 500, acc_train))

            except KeyboardInterrupt:
                print("Interrupted")
                return self

        return self

    def predict_proba(self, X):
        return self.sess.run(self.probs, feed_dict={self.X: self.pred_(X)})

    def predict(self, X):
        return self.sess.run(self.output, feed_dict={self.X: self.pred_(X)})

if __name__ == '__main__':
    from sklearn.metrics import log_loss
    from data.import_data import import_data
    from Predictors.sklearnclassifier import show_stats
    from data.numbered_authors import NumberAuthorsTransformer
    from data.pipelines import tokenize_pipe, lower_pipe, stem_pipe, lemmatize_pipe

    tr, te = import_data()

    author_enc = NumberAuthorsTransformer()

    y_train = author_enc.fit_transform(tr.author)
    y_test = author_enc.transform(te.author)

    dnn_args = {'epochs':250, 'layers':[100], 'window':5, 'pte':True, 'index_out':False}
    prob_args = {'epochs':1500, 'layers':[], 'beta_method':True, 'stem':True, 'lemma':True, 'index_out':False}

    myc = DNNEnsemble(dnn_args, prob_args)

    myc.fit(tr.text, y_train)

    pred = myc.predict(te.text)
    predict_proba = myc.predict_proba(te.text)

    acc = accuracy_score(y_test, pred)
    loss = log_loss(y_test, predict_proba)
    print("\n\nAccuracy={}".format(acc))
    print("Loss = {}".format(loss))

