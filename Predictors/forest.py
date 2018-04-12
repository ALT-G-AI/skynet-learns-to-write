from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from Predictors.sklearn import SklearnClassifier, test_sklearnclassifier, random_search_params
from scipy.stats import expon


class ForestClassifier(SklearnClassifier):
    def init_clf_(self, **kwargs):
        self.clf = RandomForestClassifier(
        #    n_jobs=-1,
            **kwargs)

if __name__ == '__main__':
    # numbered authors
    # test_sklearnclassifier(ForestClassifier, n_jobs=-1, n_estimators = 437) # 54% accuracy

    # one-hot encoded authors
    test_sklearnclassifier(ForestClassifier, AuthorProc=LabelBinarizer, n_jobs=-1, n_estimators = 152) # 19% accuracy, 28% F1

    param_dist = {
        'n_estimators': expon(scale=500),
        }

    # numbered authors
    # random_search_params(ForestClassifier, param_dist)

    # one-hot encoded authors
    #random_search_params(ForestClassifier, param_dist, AuthorProc=LabelBinarizer)
