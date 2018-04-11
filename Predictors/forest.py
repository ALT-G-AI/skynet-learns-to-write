from sklearn.ensemble import RandomForestClassifier
from Predictors.sklearn import SklearnClassifier, test_sklearnclassifier, random_search_params
from scipy.stats import expon


class ForestClassifier(SklearnClassifier):
    def init_clf_(self, **kwargs):
        self.clf = RandomForestClassifier(
        #    n_jobs=-1,
            **kwargs)

if __name__ == '__main__':
    test_sklearnclassifier(ForestClassifier, n_jobs=-1, n_estimators = 437)
    param_dist = {
        'n_estimators': expon(scale=500),
        }

    #random_search_params(ForestClassifier, param_dist)
