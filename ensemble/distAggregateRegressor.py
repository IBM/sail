"""
Regressor for ensemble models
Based on: https://github.com/yzhao062/combo/blob/master/combo/models/base.py extended for
parallelizable incremental and batch learning algorithms.
"""
import typing
from river import base
from river import optim
from .base import BaseAggregator
from river.compat import convert_river_to_sklearn
import ray
import numpy as np
from sail.utils.ray_utils import _model_fit


__all__ = ["DistAggregateRegressor"]


class DistAggregateRegressor(BaseAggregator):

    def __init__(
        self,
        estimators,
        fitted_estimators = [],
        loss = None,
        learning_rate=0.5,
        aggregator="simple"
    ):
        """
        :param estimators: Estimator objects with partial_fit defined
                    (incremental learning models).
        :param fitted_estimators: Batch learning estimators.
                Used only for scoring.
        :param aggregator: Type of aggregator. Options are "simple", "windsor" and "trim"
        """
        # if len(estimators) < 2:
        #     raise NotEnoughModels(n_expected=2, n_obtained=len(estimators))

        self.loss = optim.losses.Squared() if loss is None else loss
        self.learning_rate = learning_rate
        self.weights = [1.0] * len(estimators)
        self.aggregator = aggregator
        estimators = [convert_river_to_sklearn(est) if "river" in est.__module__ else est
                      for est in estimators]
        # self.estimators = [est.remote() for est in estimators]
        super().__init__(base_estimators=estimators, fitted_estimators=fitted_estimators)

    def _partial_fit(self, X, y=None, **kwargs):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :return:
        """
        obj_ids = []

        for i, estimator in enumerate(self.base_estimators):
            estimator.n_features_in_ = X.shape[1]
            X_id = ray.put(X)
            obj_ids.append(_model_fit.remote(estimator, X_id, y))
        self.base_estimators = ray.get(obj_ids)
        return self

    def partial_fit(self, X, y=None):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :return:
        """
        self._partial_fit(X, y)
        return self

    def predict(self, X):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :return: numpy array of shape (n_samples,).
             Predictions for input samples.
        """
        X_ensemble = []
        self.estimators = self.base_estimators
        self.estimators.extend(self.fitted_estimators)
        X_ensemble.append([model.predict(X) for model in self.estimators])
        X_ensemble = np.array(X_ensemble)
        # simple
        if self.aggregator == "simple":
            y_pred = np.nanmean(X_ensemble, axis=1)[0]
        # Windsorized
        if self.aggregator == "windsor":
            y_pred = ((np.sum(X_ensemble, axis=1) - np.amax(X_ensemble, axis=1) - np.amin(X_ensemble, axis=1)
                       + np.sort(X_ensemble)[:, 1] + np.sort(X_ensemble)[:, -2]) / (
                    X_ensemble.shape[1]))[0]
        # Trimmed
        if self.aggregator == "trim":
            y_pred = ((np.sum(X_ensemble, axis=1) - np.amax(X_ensemble, axis=1) -
                      np.amin(X_ensemble, axis=1)) / (X_ensemble.shape[1] - 2))[0]
        return y_pred