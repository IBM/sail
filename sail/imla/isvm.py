# https://github.com/Jung-Sanghyun/Incremental-SVM/blob/main/HMISVM.py
# https://github.com/galrettig/PegasosSvm/blob/master/Pegasos_Svm_Explore.ipynb
import numpy as np


class ISVM:
    def __init__(self, lam=0.001, T=10000):
        self.lam = lam
        self.T = T
        self.W = None

    def fit(self, X, y):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
                   Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :return: self
        """
        return self.partial_fit(X, y)

    def predict(self, X):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
           Input samples.
        :return: numpy.ndarray of shape (n_samples)
            Predictions for the samples in X.
        """
        ret = []
        X = np.c_[X, np.ones(X.shape[0])]
        for x in range(0, X.shape[0]):
            if self.W[-1] @ X[x] >= 0:
                ret.append(1)
            else:
                ret.append(-1)
        return np.array(ret)

    def get_params(self, deep=False):
        return {"T": self.T, "lam": self.lam}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def partial_fit(self, X, y, **kwargs):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
                   Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :return: self
        """
        y[y == 0] = -1
        X = np.c_[X, np.ones(X.shape[0])]
        self.W = np.zeros((self.T + 1, X.shape[1]))
        for t in range(0, self.T):
            ith = np.random.randint(0, X.shape[0])
            lr = 1.0 / (self.lam * (t + 1))
            w = self.W[t]
            dot = w @ X[ith]
            test = y[ith]*dot
            if y[ith] * dot < 1:
                self.W[t + 1] = (1 - (lr * self.lam)) * w + lr * y[ith] * X[ith]
            else:
                self.W[t + 1] = (1 - (lr * self.lam)) * w
        return self
