from sklearn.svm import LinearSVC

from Predictors.sklearnclassifier import SklearnClassifier, test_sklearnclassifier


class SvmClassifier(SklearnClassifier):
    def init_clf_(self, **kwargs):
        # LinearSVC is much faster but you can't change the kernel
        self.clf = LinearSVC(**kwargs)


if __name__ == '__main__':
    test_sklearnclassifier(SvmClassifier, train_limit=1000)
