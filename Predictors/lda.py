""" Module that uses Latent Dirichlet Allocation to extract topics"""
#%% Train LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

from Predictors.sklearnclassifier import show_stats
from data.import_data import import_data, clean
from data.numbered_authors import NumberAuthorsTransformer
from data.padded_sentences import PaddedSentenceTransformer


def get_topics(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[topic_idx] = [feature_names[i] for i in
                             topic.argsort()[:-n_top_words - 1:-1]]
    return topics


def get_data_(train_limit, encoder_size, get_train_data=True):
    if get_train_data:
        # Training data
        data, _ = import_data()
    else:
        # Testing data
        _, data = import_data()

    text = list(data.text)
    author = list(data.author)

    label_enc = NumberAuthorsTransformer()
    data_enc = PaddedSentenceTransformer(encoder_size=encoder_size)

    if train_limit is None:
        train_limit = len(text)
    y = label_enc.fit_transform(author[:train_limit])

    print(data_enc)
    X = data_enc.fit_transform(text[:train_limit])

    return X, y


def run_svm(train_limit=None, labels_enc_with_data=False):
    myc = LinearSVC()
    print("Getting data")
    #%% Get data
    X_train, y_train = get_data_(train_limit, 100)

    print("Training")
    #%% Fit data
    myc.fit(X_train, y_train)
    print("Getting test data")
    X_test, y_test = get_data_(train_limit, 100)
    print("Estimating")
    y_train_pred = myc.predict()
    show_stats(y_test, y_train_pred)


def main():
    """ Use Latent Dirichlet Allocation for topic allocation """
    tr, te = import_data()


    cleaned = [clean(doc) for doc in tr.text]
    #
    # authors = np.array(tr.author)
    #
    # trans = LatentDirichletAllocation(n_jobs=-2, verbose=2)
    #
    # # print(tr[:100])
    # # Topic count: determined by the KL Divergence score.
    #
    #
    # result = trans.fit_transform(tr.text)
    #
    # for a, b in zip(authors, result):
    #     assert (a == b)
    # These two values are chosen by trial and error.
    # Number of topics to categorise
    num_topics = 15
    # Number of words in each topic
    num_topic_words = 10

    # Use the standard counting vectorizer.
    tf_vectorizer = CountVectorizer(  # max_df=0.95, min_df=2,
        # max_features=3000,
        stop_words='english')
    tf_vectors = tf_vectorizer.fit_transform(cleaned)
    # Produces an array of topic->word mappings



    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf_vectors)

    run_svm()
    tf_feature_names = tf_vectorizer.get_feature_names()

    print("Running cross-validation...")

    topics = get_topics(lda, tf_feature_names, num_topic_words)
    doc_topic_distrib = lda.transform(tf_vectors)



if __name__ == "__main__":
    main()
