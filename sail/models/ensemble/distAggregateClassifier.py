"""
Classifier for ensemble models
Based on: https://github.com/yzhao062/combo/blob/master/combo/models/base.py extended for
parallelizable, incremental and batch learning algorithms.
"""

import typing
from river import base
from river import optim
from sail.models.ensemble.base import BaseAggregator
from river.compat import convert_river_to_sklearn
import ray
import numpy as np
from sail.utils.ray_utils import _model_fit

__all__ = ["DistAggregateClassifier"]


class DistAggregateClassifier(BaseAggregator):
    def __init__(
        self,
        estimators: typing.List[base.Classifier],
        fitted_estimators: typing.List[base.Classifier] = [],
        learning_rate=0.5,
        aggregator="maximization",
    ):
        """
        :param estimators: Estimator objects with partial_fit defined
                    (incremental learning models).
        :param fitted_estimators: Batch learning estimators.
                Used only for scoring.
        :param aggregator: Type of aggregator. Options are "maximization" and "majority_vote"
        """
        # if len(estimators) < 2:
        #     raise NotEnoughModels(n_expected=2, n_obtained=len(regressors))

        self.learning_rate = learning_rate
        self.weights = [1.0] * len(estimators)
        self.aggregator = aggregator
        estimators = [
            convert_river_to_sklearn(est) if "river" in est.__module__ else est
            for est in estimators
        ]
        super().__init__(
            base_estimators=estimators, fitted_estimators=fitted_estimators
        )

    def _partial_fit(self, X, y=None, classes=None, **kwargs):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :param classes: numpy.ndarray, optional.
               Unique classes in the data y.
        :return:
        """
        obj_ids = []
        for i, estimator in enumerate(self.base_estimators):
            estimator.n_features_in_ = X.shape[1]
            X_id = ray.put(X)
            obj_ids.append(_model_fit.remote(estimator, X_id, y, classes))
        estimators = ray.get(obj_ids)
        self.base_estimators = estimators
        return self

    def partial_fit(self, X, y=None, classes=None):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :param classes: numpy.ndarray, optional.
               Unique classes in the data y.
        :return:
        """
        self._partial_fit(X, y, classes)
        return self

    def predict(self, X):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :return: numpy array of shape (n_samples,).
             Class labels/predictions for input samples.
        """
        X_ensemble = []
        estimators = self.base_estimators
        estimators.extend(self.fitted_estimators)
        for i, estimator in enumerate(estimators):
            estimator.n_features_in_ = X.shape[1]
            X_ensemble.append(estimator.predict(X))
        X_ensemble = np.array(X_ensemble)

        # maximization
        if self.aggregator == "maximization":
            y_pred = np.max(X_ensemble, axis=1).ravel()
        # majority vote
        if self.aggregator == "majority_vote":
            n_samples, n_estimators = X_ensemble.shape[0], X_ensemble.shape[1]
            vote_results = np.zeros(
                [
                    n_samples,
                ]
            )
            for i in range(n_samples):
                values, counts = np.unique(X_ensemble[i], return_counts=True)
                ind = np.argmax(counts)
                vote_results[i] = values[ind]
            y_pred = vote_results.ravel()
            y_pred = np.array([int(i) for i in y_pred])
        # # average
        # if self.aggregator == "average":
        #     y_pred = np.nanmean(X_ensemble, axis=1).ravel()

        return y_pred
