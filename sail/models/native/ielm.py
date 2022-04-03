import numpy as np
from numpy.linalg import pinv
import pandas as pd
from sail.models.native.base import BaseEstimator

# Code and algorithms used as reference to convert to sklearn/sail template:
# https://medium.datadriveninvestor.com/extreme-learning-machine-for-simple-classification-e776ad797a3c
# https://github.com/chickenbestlover/Online-Recurrent-Extreme-Learning-Machine/blob/master/timeseries_prediction.ipynb
# https://github.com/khushalpt/Incremental-Extreme-Machine-Learning/blob/main/IELM_Regression_UCI.ipynb
# https://github.com/5663015/elm/blob/master/elm.py


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(0, x)


def _linear(X, weights, bias):
    return np.dot(X, np.transpose(weights)) + bias


class IELM(BaseEstimator):
    def __init__(
        self,
        numInputNodes,
        numOutputNodes,
        numHiddenNodes,
        forgettingFactor=0.999,
        lambda_value=0.01,
    ):
        """
        :param numInputNodes: int. Number of input nodes.
        :param numOutputNodes: int. Number of output nodes.
        :param numHiddenNodes: int. Number of hidden nodes.
        :param forgettingFactor: float, 0<=forgettingFactor<=1. Forgetting factor for old samples.
        :param lambda_value: float. Regularization factor.
        """

        self._inputNodes = numInputNodes
        self._outputNodes = numOutputNodes
        self._numHiddenNodes = numHiddenNodes

        self._weights = np.random.random((self._numHiddenNodes, self._inputNodes))
        self._weights = self._weights * 2 - 1

        self._bias = np.random.random((1, self._numHiddenNodes)) * 2 - 1

        self._beta = np.zeros([self._numHiddenNodes, self._outputNodes])

        self._M = np.linalg.inv(lambda_value * np.eye(self._numHiddenNodes))

        self.forgettingFactor = forgettingFactor

    def calculateHiddenLayerActivation(self, X):
        """
        :param X : numpy array of shape (n_samples, n_features)
            The input samples.
        :return: numpy array of shape (n_samples, n_features).
             Transformed sample
        """
        V = _linear(X, self._weights, self._bias)
        # H = sigmoidActFunc(V)
        H = _relu(V)
        return H

    def fit(self, X, y):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
                   Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :return: self
        """
        # if isinstance(y, pd.Series):
        #     y = (y.values).reshape(-1)
        numSamples = None
        numOutputs = None
        if y.ndim == 1:
            y = y.to_numpy()  # np.reshape(y.values,-1)
            y = y.reshape((y.shape[0], 1))
            (numSamples, numOutputs) = (y.shape[0], 1)
        else:
            (numSamples, numOutputs) = y.shape
        assert X.shape[0] == y.shape[0]

        H = self.calculateHiddenLayerActivation(X)
        Ht = np.transpose(H)

        self._M = (1 / self.forgettingFactor) * self._M - np.dot(
            (1 / self.forgettingFactor) * self._M,
            np.dot(
                Ht,
                np.dot(
                    pinv(
                        np.eye(numSamples)
                        + np.dot(H, np.dot((1 / self.forgettingFactor) * self._M, Ht))
                    ),
                    np.dot(H, (1 / self.forgettingFactor) * self._M),
                ),
            ),
        )

        self._beta = self._beta + np.dot(self._M, np.dot(Ht, y - np.dot(H, self._beta)))
        return self

    def predict(self, X):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
           Input samples.
        :return: numpy.ndarray of shape (n_samples)
            Predictions for the samples in X.
        """
        H = self.calculateHiddenLayerActivation(X)
        prediction = np.dot(H, self._beta)
        return prediction.ravel()

    def partial_fit(self, X, y, **kwargs):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
                   Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :return: self
        """
        return self.fit(X, y)
