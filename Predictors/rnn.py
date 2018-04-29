from sklearn.base import BaseEstimator, ClassifierMixin
from math import floor
import tensorflow as tf
import numpy as np


class RNNClassifier(BaseEstimator, ClassifierMixin):

    """
    Generic RNN - can use ordinary RNN, LSTM or GRU
    
    Internal state is fed into a fully connected logistic regression layer and then softmax is applied to achieve classification
       Basically the logistic regression layer provides a probability for each class.
    
    Based upon "Training a Sequence Classifier" on page 587 of "Hands on Machine Learning with Scikit-Learn and TensorFlow" by Aurelien Geron
    """

    def __init__(self, CellType=tf.contrib.rnn.GRUCell, n_neurons=100, learning_rate=0.001, n_outputs=4,
                 n_out_neurons=10, n_inputs=50, n_steps=5, n_epochs=100, batch_size=100, **cell_args):
        """
        n_inputs = word encoding size
        n_steps = amount of recursion e.g. window size
        n_neurons = number of neurons in RNN
        n_out_neurons = number of neurons in the fully connected output layer (= the number of classes)
        n_outputs = number of outputs
        """
        self.CellType = CellType
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs
        self.n_out_neurons = n_out_neurons
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cell_args = cell_args

        self.construct_()

    def construct_(self):
        """
        tensorflow construction stage
        """

        self.X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        #print("\n\nExpecting X.shape = {}, dtype = {}\n".format(self.X.shape, self.X.dtype))
        #y = tf.placeholder(tf.int64, [None, self.n_outputs])
        self.y = tf.placeholder(tf.int64, [None])
        #print("\n\nExpecting y.shape = {}, dtype = {}\n".format(self.y.shape, self.y.dtype))

        cell = self.CellType(num_units=self.n_neurons, **self.cell_args)
        outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)

        out_layer = tf.layers.dense(states, self.n_out_neurons)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=out_layer)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.probs = tf.nn.softmax(out_layer)
        _, self.output = tf.nn.top_k(out_layer, k=1)

        self.training_op = optimizer.minimize(loss)
        self.correct = tf.nn.in_top_k(out_layer, self.y, 1)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        self.init = tf.global_variables_initializer()


    def fit(self, sentences, labels, sess=None):
        """
        sentences and labels should already be encoded
        """

        if sess is None:
            sess = tf.get_default_session()

        test_prop = 0.1
        train_prop = 1.0 - test_prop
        split = floor(train_prop*sentences.shape[0])

        X_train = sentences[:split]
        X_test = sentences[split:]
        y_train = labels[:split]
        y_test = labels[split:]

        print("Training RNN Classifier")

        sess.run(self.init)
        for epoch in range(self.n_epochs):
            try:
                for iteration in range(sentences.shape[0] // self.batch_size):
                    # pick a random batch
                    batch_choices = np.random.choice(X_train.shape[0], self.batch_size, replace=True)
                    X_batch = X_train[batch_choices]
                    y_batch = y_train[batch_choices]
                    #print("type(X_batch = {}, type(y_batch) = {}\n".format(type(X_batch), type(y_batch)))
                    #print("X_batch.shape = {}, y_batch.shape = {}\n".format(X_batch.shape, y_batch.shape))
                    #print("X_batch.dtype = {}, y_batch.dtype = {}\n".format(X_batch.dtype, y_batch.dtype))

                    sess.run(self.training_op, feed_dict = { self.X : X_batch , self.y : y_batch })

                acc_train = self.accuracy.eval(feed_dict={self.X: X_batch, self.y: y_batch})
                acc_test =  self.accuracy.eval(feed_dict={self.X: X_test, self.y: y_test})
                print("{}/{}:\t\tTrain accuracy:\t{:.3f}\tTest accuracy:\t{:.3f}".format(epoch, self.n_epochs, acc_train, acc_test))

            except KeyboardInterrupt: # stop early with Control+C (SIGINT)
                print("\nInterrupted by user at epoch {}/{}".format(epoch, self.n_epochs))
                return self

        return self

    def predict_proba(self, X, sess=None):
        """
        probabillites for each class
        """
        if sess is None:
            sess = tf.get_default_session()

        self.init.run()
        return sess.run(self.probs, feed_dict={self.X: X})

    def predict(self, X, sess=None):
        """
        predict class
        """
        if sess is None:
            sess = tf.get_default_session()

        out = sess.run(self.output, feed_dict={self.X: X})
        return out 

if __name__ == '__main__':
    from data.numbered_authors import NumberAuthorsTransformer
    from data.windowed_sentences import WindowedSentenceTransformer
    from data.padded_sentences import PaddedSentenceTransformer
    from data.import_data import import_data
    from Predictors.sklearn import show_stats
    
    tr, te = import_data()
    #data_enc = WindowedSentenceTransformer(encoder_size=50)
    data_enc = PaddedSentenceTransformer(encoder_size=50)
    label_enc = NumberAuthorsTransformer()

    y_train = label_enc.fit_transform(list(tr.author))
    y_test = label_enc.transform(list(te.author))
    
    # for WindowedSentenceTransformer
    #X_train, y_train = zip(*data_enc.fit_transform(tr.text, y_train))
    #X_test, y_test = zip(*data_enc.transform(te.text, y_test))

    # for PaddedSentenceTransformer
    X_train = data_enc.fit_transform(tr.text)
    X_test = data_enc.transform(te.text)

    sess = tf.Session()
    with tf.Session() as sess:
        with sess.as_default():
            # about 40% accuracy. These two are for WindowedSentenceTransformer
            # rnn = RNNClassifier(CellType = tf.contrib.rnn.LSTMCell, n_out_neurons=3, n_outputs=1, n_epochs=200, n_neurons=200,
            #                    state_is_tuple=False) # TODO state_is_tuple is deprecated
            #rnn = RNNClassifier(CellType = tf.contrib.rnn.GRUCell, n_out_neurons=3, n_outputs=1, n_epochs=200, n_neurons=200)

            # For using PaddedSentenceTransformer. 69% accuracy on the test set but 100% from early epochs on the training set
            rnn = RNNClassifier(n_steps=50, CellType = tf.contrib.rnn.GRUCell, n_out_neurons=3, n_outputs=1, n_epochs=200, n_neurons=200)

            rnn.fit(np.array(X_train), np.array(y_train))

            y_test_pred = rnn.predict(X_test)

    show_stats(y_test, y_test_pred)
