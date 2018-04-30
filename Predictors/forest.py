from scipy.stats import expon
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer

from Predictors.sklearnclassifier import SklearnClassifier, test_sklearnclassifier, random_search_params
from data.windowed_sentences import WindowedSentenceTransformer


class ForestClassifier(SklearnClassifier):
    def init_clf_(self, **kwargs):
        self.clf = RandomForestClassifier(
            #    n_jobs=-1,
            **kwargs)


if __name__ == '__main__':
    # numbered authors
    # test_sklearnclassifier(ForestClassifier, n_jobs=-1, n_estimators = 437) # 54% accuracy

    # one-hot encoded authors
    # test_sklearnclassifier(ForestClassifier, AuthorProc=LabelBinarizer, n_jobs=-1, n_estimators = 152) # 19% accuracy, 28% F1

    # windowed, numbered authors
    test_sklearnclassifier(ForestClassifier, DataProc=WindowedSentenceTransformer,
                           labels_enc_with_data=True, n_jobs=-1, n_estimators=437)  # 41% accuracy

    param_dist = {
        'n_estimators': expon(scale=500),
    }

    # numbered authors
    # random_search_params(ForestClassifier, param_dist)

    # one-hot encoded authors
    # random_search_params(ForestClassifier, param_dist, AuthorProc=LabelBinarizer)

    # windowed, numbered authors
    random_search_params(ForestClassifier, param_dist, DataProc=WindowedSentenceTransformer,
                         labels_enc_with_data=True)
