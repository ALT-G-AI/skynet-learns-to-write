from math import floor

import sys
import numpy as np
import tensorflow as tf
import datetime
from sklearn.base import BaseEstimator, ClassifierMixin


class RNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Generic RNN - can use ordinary RNN, LSTM or GRU
    
    Internal state is fed into a fully connected logistic regression layer and then softmax is applied to achieve classification
       Basically the logistic regression layer provides a probability for each class.
    
    Based upon "Training a Sequence Classifier" on page 587 of "Hands on Machine Learning with Scikit-Learn and TensorFlow" by Aurelien Geron
    """

    def __init__(self, cell_type=tf.contrib.rnn.GRUCell, n_neurons=30, learning_rate=0.001, n_outputs=1,
                 n_out_neurons=3, n_inputs=50, n_steps=50, n_epochs=20, batch_size=200, dropout_rate=0.0,
                 he_init=False, **cell_args):
        """
        cell_type = The class used to instantiate the cell. This allows us to choose between LSTM, GRU and BasicRNN
        n_neurons = The number of neurons in the cell_type instance
        learning_rate = The learning rate used with the optimiser
        n_outputs = number of outputs
        n_out_neurons = number of neurons in the fully connected output layer (= the number of classes)
        n_inputs = word encoding size
        n_steps = amount of recursion e.g. window size
        n_epochs = the number of epochs to train for
        batch_size = the number of training examples used in each batch during training
        dropout_rate = proportion of dropout: 0 being none and 1 being all
        he_init = do we use He Initialisation
        """
        self.cell_type = cell_type
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs
        self.n_out_neurons = n_out_neurons
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cell_args = cell_args
        self.dropout = (dropout_rate != 0)
        self.dropout_rate = dropout_rate
        self.he_init = he_init

        # each RNNClassifier needs its own graph so that creating one after another for grid search doesn't
        # create naming conflicts in the tensorflow graph. Sessions are subordinate to graphs so recreating
        # the session isn't sufficient. 
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()

        self.construct_()

    def construct_(self):
        """
        tensorflow construction stage
        """
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
            self.y = tf.placeholder(tf.int64, [None])

            if self.he_init:
                he_init = tf.contrib.layers.variance_scaling_initializer()
                basic_cell = self.cell_type(num_units=self.n_neurons, kernel_initializer=he_init, **self.cell_args)
            else:
                basic_cell = self.cell_type(num_units=self.n_neurons, **self.cell_args)

            if self.dropout:
                cell = tf.contrib.rnn.DropoutWrapper(basic_cell, input_keep_prob = 1.0 - self.dropout_rate,
                                                     output_keep_prob= 1.0 - self.dropout_rate)
            else:
                cell = basic_cell

            outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)

            if self.he_init:
                out_layer = tf.layers.dense(states, self.n_out_neurons, kernel_initializer=he_init)
            else:
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

            # group the stuff we care about so it is easier to restore them
            for operation in (self.X, self.y, self.probs, self.output, self.training_op, self.correct, self.accuracy, self.init):
                tf.add_to_collection("rnn_members", operation)

    def __del__(self):
        try:
            self.sess.close()
        except AttributeError: # if something breaks before we get chance to make self.sess
            pass

    def fit(self, sentences, labels):
        """
        sentences and labels should already be encoded
        """
        test_prop = 0.05 # this is only to display progress so setting this too high just wastes data
        train_prop = 1.0 - test_prop
        split = floor(train_prop * sentences.shape[0])

        sentences = np.array(sentences)
        labels = np.array(labels)

        X_train = sentences[:split]
        X_test = sentences[split:]
        y_train = labels[:split]
        y_test = labels[split:]

        print("Training RNN Classifier")

        with self.graph.as_default():
            self.sess.run(self.init)
            for epoch in range(self.n_epochs):
                try:
                    for iteration in range(sentences.shape[0] // self.batch_size):
                        # pick a random batch
                        batch_choices = np.random.choice(X_train.shape[0], self.batch_size, replace=True)
                        X_batch = X_train[batch_choices]
                        y_batch = y_train[batch_choices]

                        self.sess.run(self.training_op, feed_dict={self.X: X_batch, self.y: y_batch})

                    acc_train = self.sess.run(self.accuracy, feed_dict={self.X: X_batch, self.y: y_batch})
                    acc_test = self.sess.run(self.accuracy, feed_dict={self.X: X_test, self.y: y_test})
                    print(
                        "{}/{}:\t\tTrain accuracy:\t{:.3f}\tTest accuracy:\t{:.3f}".format(epoch+1, self.n_epochs, acc_train,
                                                                                        acc_test))
                    sys.stdout.flush()

                except KeyboardInterrupt:  # stop early with Control+C (SIGINT)
                    print("\nInterrupted by user at epoch {}/{}".format(epoch, self.n_epochs))
                    return self
        return self

    def predict_proba(self, X):
        """
        probabilities for each class
        """
        with self.graph.as_default():
            self.sess.run(self.init)
            return self.sess.run(self.probs, feed_dict={self.X: X})

    def predict(self, X):
        """
        predict class
        """
        with self.graph.as_default():
            out = self.sess.run(self.output, feed_dict={self.X: X})
            return out

    def save(self, path=None):
        """
        Save the model to the disk
        """

        if path is None:
            path = "./RNN-" + datetime.datetime.now().isoformat() + ".ckpt"

        print("Saving to {}".format(path))
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, path)

    def restore(self, path):
        """
        restore from a saved model

        Don't try restoring into an instance with different parameters. This may not work.
        You will also need to save and restore the data and label encoders which the model was trained with.
        """
        print("Restoring from {}".format(path))

        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, path)

            self.X, self.y, self.probs, self.output, self.training_op, self.correct, self.accuracy, self.init \
                = tf.get_collection("rnn_members")

if __name__ == '__main__':
    from data.numbered_authors import NumberAuthorsTransformer
    from data.windowed_sentences import WindowedSentenceTransformer
    from data.padded_sentences import PaddedSentenceTransformer
    from data.import_data import import_data
    from Predictors.sklearnclassifier import show_stats
    from sklearn.metrics import accuracy_score, log_loss

    # padded sentences vs windows
    USE_PADDED_SENTENCES = True

    tr, te = import_data()

    if USE_PADDED_SENTENCES:
        data_enc = PaddedSentenceTransformer(encoder_size=50)
    else:
        data_enc = WindowedSentenceTransformer(encoder_size=50)

    label_enc = NumberAuthorsTransformer()

    y_train = label_enc.fit_transform(list(tr.author))
    y_test = label_enc.transform(list(te.author))

    if USE_PADDED_SENTENCES:
        X_train = data_enc.fit_transform(tr.text)
        X_test = data_enc.transform(te.text)
    else:
        X_train, y_train = zip(*data_enc.fit_transform(tr.text, y_train))
        X_test, y_test = zip(*data_enc.transform(te.text, y_test))

    PARAM_SEARCH = False
    if PARAM_SEARCH:
        print("Parameter grid search. This will take over and hour")
        # I was unable to get the sklearn parameter search working.
        # sklearn does not give the correct parameters to __init__ and seems to attempt to change them later
        # this is a crude grid search

        # params we are using every time
        always_params = {'n_steps': 50, 'n_out_neurons': 3, 'n_outputs': 1, 'n_epochs': 200}

        # params we are changing
        n_neurons_grid = range(20, 200, 10)
        cell_types_grid = (tf.contrib.rnn.GRUCell, tf.contrib.rnn.LSTMCell)
        dropout_rate_grid = (0.0, 0.1, 0.2, 0.3, 0.4)
        learning_rate_grid = (0.0001, 0.001, 0.01)
        he_init_grid = (False,)

        search = [{**always_params.copy(), 'n_neurons': nn, 'cell_type': ct, 'dropout_rate': dr, 'learning_rate': lr,
                   'he_init': he} for nn in n_neurons_grid for ct in cell_types_grid for dr in dropout_rate_grid
                  for lr in learning_rate_grid for he in he_init_grid]

        results = []

        for params in search:
            # makes reading the state out of the middle of the LSTM work
            if params['cell_type'] == tf.contrib.rnn.LSTMCell:
                params['state_is_tuple'] = False

            rnn = RNNClassifier(**params)
            rnn.fit(X_train, y_train)
            y_test_pred = rnn.predict(X_test)
            print("\nTesting params {}".format(params))
            show_stats(y_test, y_test_pred)
            print('\n')
            sys.stdout.flush()

            results.append((accuracy_score(y_test, y_test_pred), params))

        print("\n\n" + "="*24)
        print("\tRESULTS")
        print("="*24)
        for a, p in results:
            p_display = [(k, p[k]) for k in ('n_neurons', 'cell_type')]
            print("{:.3f}\t{}".format(a, p_display))

        acc, params = max(results)
        params_display = [(k, params[k]) for k in ('n_neurons', 'cell_type')]
        print("\nBest was\n{:.3f} \t{}".format(acc, params_display))

    else:
        print("Old test")
        # about 40% accuracy. These two are for WindowedSentenceTransformer
        # rnn = RNNClassifier(cell_type = tf.contrib.rnn.LSTMCell, n_out_neurons=3, n_outputs=1, n_epochs=200, n_neurons=200,
        #                    state_is_tuple=False) # TODO state_is_tuple is deprecated
        # rnn = RNNClassifier(cell_type = tf.contrib.rnn.GRUCell, n_out_neurons=3, n_outputs=1, n_epochs=200, n_neurons=200)

        # For using PaddedSentenceTransformer. 73% accuracy on the test set 
        rnn_params = {'n_steps': 50, 'cell_type': tf.contrib.rnn.LSTMCell, 'n_out_neurons': 3, 'n_outputs': 1,
                      'n_epochs': 200, 'n_neurons': 20, 'dropout_rate': 0.00, 'learning_rate': 0.001, 'state_is_tuple': False}
        rnn = RNNClassifier(**rnn_params)

        rnn.fit(X_train, y_train)

        PATH = "./rnn_save_test.ckpt"
        rnn.save(path=PATH)

        y_test_pred = rnn.predict(X_test)
        y_test_probs = rnn.predict_proba(X_test)

        show_stats(y_test, y_test_pred)
        print("log loss: {}".format(log_loss(y_test, y_test_probs)))

        # test restoring
        tf.reset_default_graph()

        with tf.Session() as sess:
            with sess.as_default():
                rnn_restored = RNNClassifier(**rnn_params)
                rnn_restored.restore(PATH)
                y_test_pred = rnn_restored.predict(X_test)

        show_stats(y_test, y_test_pred) # observe that these stats are the same as for the original rnn

