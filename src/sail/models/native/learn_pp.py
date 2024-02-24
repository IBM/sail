from sklearn.linear_model import SGDClassifier
from sail.models.sklearn.naive_bayes import GaussianNB 

import numpy as np

class LearnPPClassifier:
    def __init__(self, base_classifier=GaussianNB, n_estimators=10):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.classifiers = []
        self.weights = []

    def fit(self, X, y, classes=None):
        for _ in range(self.n_estimators):
            base_clf = self.base_classifier()
            self.classifiers.append(base_clf)
            base_clf.partial_fit(X, y, classes=np.unique(y))
            y_pred = base_clf.predict(X)
            error_rate = np.sum(y_pred != y) / len(y)
            weight = np.log((1 - error_rate + 1e-10) / (error_rate + 1e-10))
            self.weights.append(weight)

    def partial_fit(self, X, y, classes=None):
        if len(self.weights)==0:
            self.fit(X,y)
        else:
            for i, clf in enumerate(self.classifiers):
                clf.partial_fit(X, y, classes=np.unique(y))
                y_pred = clf.predict(X)
                error_rate = np.sum(y_pred != y) / len(y)
                weight = np.log((1 - error_rate + 1e-10) / (error_rate + 1e-10))
                self.weights[i] += weight

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            predictions[:, i] = clf.predict(X)

        weighted_predictions = np.dot(predictions, self.weights)
        sigmoid_predictions = 1 / (1 + np.exp(-weighted_predictions))
        return np.round(sigmoid_predictions)