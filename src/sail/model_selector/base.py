from sail.utils.ray_utils import _model_fit
import ray
from sklearn.metrics import r2_score
from abc import ABC, abstractmethod
from sail.models.native.base import MetaEstimatorMixin

# Reference: https://github.com/AlexImb/automl-streams
# https://github.com/yzhao062/combo/blob/master/combo/models/base.py


class ModelSelectorBase(ABC, MetaEstimatorMixin):
    """
    Base class for distributed model selection.
    """

    def __init__(self, estimators, fitted_estimators=[], metrics=r2_score):
        """
        :param estimators: Estimator objects with partial_fit defined
                    (incremental learning models).
        :param fitted_estimators: Batch learning estimators.
                Used only for scoring.
        :param metrics: Object that defines the quality of predictions
           (ex. metrics.accuracy_score in scikit-learn)
        """
        self.base_estimators = estimators  # [e.__class__() for e in estimators]
        self.fitted_estimators = fitted_estimators
        self.best_model_index = 0
        self.metrics = metrics

    @abstractmethod
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
        raise NotImplementedError

    def _partial_fit_estimators(self, X, y, classes=None, sample_weight=None):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :param classes: numpy.ndarray, optional.
               Unique classes in the data y.
        :param sample_weight: User defined weights for base estimators [Not implemented yet]
        :return: None
        """
        trained_models = []
        for estimator in self.base_estimators:
            estimator.n_features_in_ = X.shape[1]
            trained_models.append(
                _model_fit.remote(model=estimator, X=X, y=y, classes=classes)
            )
        self.base_estimators = ray.get(trained_models)

    @abstractmethod
    def _get_best_model_index(self, X, y):
        pass

    def fit(self, X, y, classes=None):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :param classes: numpy.ndarray, optional.
               Unique classes in the data y.
        :return:
        """
        self.partial_fit(X, y, classes)

    def get_best_model_index(self, X, y):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels (ground truth) for the target variable.
        :return: int. Returns the index of the best model based on user defined metrics.
        """
        self.best_model_index = self._get_best_model_index(X, y)
        return self.best_model_index

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :return: numpy array of shape (n_samples,)
            The class probabilities of the input samples.
        """
        estimators = self.base_estimators
        estimators.extend(self.fitted_estimators)
        return estimators[self.best_model_index].predict_proba(X)

    def predict(self, X):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :return: numpy array of shape (n_samples,).
             Class labels/predictions for input samples.
        """
        estimators = self.base_estimators
        estimators.extend(self.fitted_estimators)
        return estimators[self.best_model_index].predict(X)

    # def reset(self):
    #     self.base_estimators = [est.reset() for est in self.base_estimators]
    #     self.fitted_estimators = [est.reset() for est in self.fitted_estimators]
    #     self.best_model_index = 0
    #     return self

    def get_best_model(self):
        """
        :return: Current best model in the list of base estimators and fitted estimators.
        """
        estimators = self.base_estimators
        estimators.extend(self.fitted_estimators)
        return estimators[self.best_model_index]
