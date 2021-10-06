import math
import typing
from river import base
from river import optim
from .base import BaseAggregator
from river.compat import convert_river_to_sklearn
import ray
import numpy as np


@ray.remote
def _incremental_learn(estimator, X, length, y_pred_mean, weight, y, learning_rate):
    # X = ray.get(X_id)
    y_pred = estimator.predict(X)
    y_pred_mean += weight * (y_pred - y_pred_mean) / length
    loss = np.average((y - y_pred) ** 2, axis=0)  # self.loss(y_true=y, y_pred=y_pred)
    weight *= math.exp(-learning_rate * loss)
    estimator.partial_fit(X, y)
    return [estimator, weight, y_pred_mean]


__all__ = ["DistEWARegressor"]


class DistEWARegressor(BaseAggregator):
    def __init__(
        self,
        estimators,
        loss= None,
        learning_rate=0.5,
    ):
        """
        :param estimators: Estimator objects with partial_fit defined
                    (incremental learning models).
        :param loss: loss function to be minimized
        :param learning_rate: The learning rate by which the model weights are multiplied at each iteration.
        """

        # if len(estimators) < 2:
        #     raise NotEnoughModels(n_expected=2, n_obtained=len(estimators))

        self.loss = optim.losses.Squared() if loss is None else loss
        self.learning_rate = learning_rate
        self.weights = [1.0] * len(estimators)

        estimators = [convert_river_to_sklearn(est) if "river" in est.__module__ else est
                      for est in estimators]
        # self.estimators = [est.remote() for est in estimators]
        super().__init__(estimators)

    def _partial_fit(self, X, y=None, **kwargs):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :param kwargs:
        :return: numpy.ndarray of shape (n_samples)
        """
        y_pred_mean = 0.0

        # Make a prediction and update the weights accordingly for each model
        length = len(self)
        total = 0
        obj_ids = []

        for i, estimator in enumerate(self.base_estimators):
            estimator.n_features_in_ = X.shape[1]
            X_id = ray.put(X)
            obj_ids.append(_incremental_learn.remote(estimator, X_id,
                                                     length, y_pred_mean, self.weights[i],
                                  y, self.learning_rate))
        ray_values = ray.get(obj_ids)
        self.base_estimators = [i[0] for i in ray_values]
        self.weights = [i[1] for i in ray_values]
        y_pred_mean = [i[2] for i in ray_values]
        y_pred_mean = np.sum(y_pred_mean)
        total = sum(self.weights)
        # Normalize the weights so that they sum up to 1
        if total:
            for i, _ in enumerate(self.weights):
                self.weights[i] /= total

        return y_pred_mean

    def partial_fit(self, X, y=None, classes=None):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :param kwargs:
        :return: numpy.ndarray of shape (n_samples)
        """

        self._partial_fit(X, y)
        return self

    def predict(self, X):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :return: numpy array of shape (n_samples,).
             Class labels/predictions for input samples.
        """
        for i, estimator in enumerate(self.base_estimators):
            if not hasattr(estimator, 'n_features_in_'):
                estimator.n_features_in_ = X.shape[1]
        return sum(
            model.predict(X) * weight for model, weight in zip(self.base_estimators, self.weights)
        )