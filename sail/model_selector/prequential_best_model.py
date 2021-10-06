from sklearn.metrics import accuracy_score
from .base import ModelSelectorBase
from river.compat.river_to_sklearn import convert_river_to_sklearn


class PrequentialBestModelSelector(ModelSelectorBase):
    def __init__(self, estimators,  fitted_estimators=[], metrics=accuracy_score):
        """
        :param estimators: Estimator objects with partial_fit defined
                    (incremental learning models).
        :param fitted_estimators: Batch learning estimators.
                Used only for scoring.
        :param metrics: Object that defines the quality of predictions
           (ex. metrics.accuracy_score in scikit-learn)
        """
        self.best_model_index = 0
        estimators = [convert_river_to_sklearn(est) if "river" in est.__module__ else est
                      for est in estimators]
        super().__init__(estimators=estimators, fitted_estimators=fitted_estimators,
                         metrics=metrics)

    def partial_fit(self, X, y=None, classes=None):
        """
        :param model: Any machine learning model with partial_fit function defined.
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :param classes: numpy.ndarray, optional.
               Unique classes in the data y.
        :return:
        """
        self.best_model_index = self._get_best_model_index(X, y)
        self._partial_fit_estimators(X, y)
        return self

    def _get_best_model_index(self, X, y):
        """
        :param X_test: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y_test: numpy.ndarray of shape (n_samples)
               Labels (ground truth) for the target variable.
        :return: int. Returns the index of the best model based on user defined metrics.
        """
        scores = []
        estimators = self.base_estimators
        estimators.extend(self.fitted_estimators)
        for estimator in estimators:
            estimator.n_features_in_ = X.shape[1]
            # scores.append(estimator.predict(X))
            scores.append(self.metrics(y, estimator.predict(X)))
        return scores.index(max(scores))
