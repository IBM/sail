import numpy as np
from numpy.linalg import pinv
import math


def forgettingFunction(num_of_samples):
    return math.exp(0.7 * num_of_samples)


def sigmoidActivation(x):
    return 1.0 / (1.0 + math.exp(-x))


def inv_sigmoid(x):
    y = 1.0 / (1.0 + math.exp(-x))
    return math.log((y / (1 - y)))


def derv_sigmoid(x):
    return 1.0 * math.exp(-x) / (1.0 + math.exp(-x)) ** 2


class RoIML_Model:
    def __init__(self, num_x_features, num_output):
        """
        :param x_features: int. (x_i) no. of features in input x also equal to no. of input nodes.
        :param num_output: int. (d_J) or no. of output neurons/nodes
        """
        self.num_xi = num_x_features
        self.num_dj = num_output

        self.A = np.zeros((num_x_features, num_x_features))
        self.b = np.zeros((num_x_features, num_output))
        self.w_ji = np.zeros((num_x_features, num_output))

    def calculate_A_pi(self, s, x_ps, x_is, d_js):
        """
        :param s: sth training sample from the input data
        :param i: number of input features for a given training sample
        :param X: numpy array of shape (num_samples, num_x_features) The input samples.
        :param d: numpy.ndarray of shape (num_samples, j)
               Labels for the target variable/class.
        """

        h_js = forgettingFunction(s)
        # d_js = d[s][j]
        # x_ps = X[s][p]
        dhat_js = inv_sigmoid(d_js)
        df_dhat_js = derv_sigmoid(dhat_js)
        # x_is = X[s][i]
        A_pi = h_js * x_is * x_ps * df_dhat_js * df_dhat_js

        return A_pi

    def calculate_b_pj(self, s, x_ps, d_js):
        """
        :param s: sth training sample from the input data
        :param j: number of output neurons which is the number of classes for classification problems or J=1 for regression problems
        :param X: numpy array of shape (num_samples, num_x_features) The input samples.
        :param d: numpy.ndarray of shape (num_samples, j)
               Labels for the target variable/class.
        """

        # x_ps = X[s][p]
        # d_js = d[s][j]
        h_js = forgettingFunction(s)
        dhat_js = inv_sigmoid(d_js)
        df_dhat_js = derv_sigmoid(dhat_js)
        b_pj = h_js * dhat_js * x_ps * df_dhat_js * df_dhat_js

        return b_pj

    def partial_fit(self, s, X, d_js, j):
        for p in range(self.num_xi):
            x_ps = X[s][p]
            for i in range(self.num_xi):
                x_is = X[s][i]
                A_pis = self.calculate_A_pi(s, x_ps, x_is, d_js)
                self.A[p][i] = self.A[p][i] + A_pis
            b_pjs = self.calculate_b_pj(s, x_ps, d_js)
            self.b[p][j] = self.b[p][j] + b_pjs
        # calculate w_ji using A_pi and b_pj
        inv_A = np.linalg.pinv(self.A)
        self.w_ji = np.dot(inv_A, self.b)
        return self

    def predict(self, test_s):
        return np.dot(test_s, self.w_ji)
