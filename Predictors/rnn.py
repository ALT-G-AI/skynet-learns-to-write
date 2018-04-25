from sklearn.base import BaseEstimator, ClassifierMixin
from math import floor
import tensorflow as tf
import numpy as np


class RNNClassifier(BaseEstimator, ClassifierMixin):

    """
    Generic RNN - can use ordinary RNN, LSTM or GRU
    
    Internal state is fed into a fully connected logistic regression layer and then softmax is applied to achieve classification
    
    Based upon "Training a Sequence Classifier" on page 587 of "Hands on Machine Learning with Scikit-Learn and TensorFlow" by Aurelien Geron
    """

    def __init__(self, CellType=tf.contrib.rnn.GRUCell, n_neurons=100, learning_rate=0.001, n_outputs=4,
                 n_out_neurons=10, n_inputs=50, n_steps=5, n_epochs=100, batch_size=100):
        """
        n_inputs = word encoding size
        n_steps = amount of recursion e.g. window size
        n_neurons = number of neurons in RNN
        n_out_neurons = number of neurons in the fully connected output layer
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

        cell = self.CellType(num_units=self.n_neurons)
        outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)

        out_layer = tf.layers.dense(states, self.n_out_neurons)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=out_layer)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.training_op = optimizer.minimize(loss)
        self.correct = tf.nn.in_top_k(out_layer, self.y, 1)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        self.init = tf.global_variables_initializer()


    def fit(self, sentences, labels):
        """
        sentences and labels should already be encoded
        """

        test_prop = 0.1
        train_prop = 1.0 - test_prop
        split = floor(train_prop*sentences.shape[0])

        X_train = sentences[:split]
        X_test = sentences[split:]
        y_train = labels[:split]
        y_test = labels[split:]

        print("Training RNN Classifier")

        with tf.Session() as sess:
            self.init.run()
            for epoch in range(self.n_epochs):
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
                print("{}/{}:\tTrain accuracy:\t{:.3f}\tTest accuracy:\t{:.3f}".format(epoch, self.n_epochs, acc_train, acc_test))

        return self

    def predict(self, X):
        pass

if __name__ == '__main__':
    from data.numbered_authors import NumberAuthorsTransformer
    from data.windowed_sentences import WindowedSentenceTransformer
    from Predictors.sklearn import get_data_

    rnn = RNNClassifier(n_outputs=1)
    
    X_tup, y_tup = get_data_(None, NumberAuthorsTransformer, WindowedSentenceTransformer, True, 50)

    rnn.fit(np.array(X_tup), np.array(y_tup))

