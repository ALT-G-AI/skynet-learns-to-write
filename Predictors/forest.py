from sklearn.ensemble import RandomForestClassifier
from Predictors.sklearn import SklearnClassifier, test_sklearnclassifier



class ForestClassifier(SklearnClassifier):
    def init_clf_(self, **kwargs):
        self.clf = RandomForestClassifier(
            n_jobs=-1,
            **kwargs)

if __name__ == '__main__':
    test_sklearnclassifier(ForestClassifier)
