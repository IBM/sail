# # Mostly from: https://github.com/ylytkin/python-lasvm/blob/main/lasvm/lasvm.py

import numpy as np  
class LASVM:
    def __init__(self, C=1.0, tolerance=1e-3):
        self.C = C
        self.tolerance = tolerance
        self.alpha = None
        self.X = None
        self.y = 0
        self.b = 0  # Initialize self.b to zero

    def fit(self, X, y):
        y = np.where(y == 0, -1, y)
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if self.alpha is None:
            self.X = X
            self.y = y
            num_samples, num_features = X.shape
            self.alpha = np.zeros(num_samples)
        else:
            num_samples, num_features = X.shape
            num_existing_samples, _ = self.X.shape
            self.X = np.vstack([self.X, X])
            self.y = np.concatenate([self.y, y])
            
            # Extend alpha with zeros for new samples
            self.alpha = np.hstack([self.alpha, np.zeros(num_samples)])
            
            num_samples, _ = self.X.shape

        for _ in range(num_samples):
            error = self.predict(X) - y
            idx1, idx2 = self.select_two_instances(error)
            if idx1 == -1:
                break
            eta = 2.0 * self.X[idx1].dot(self.X[idx2]) - self.X[idx1].dot(self.X[idx1]) - self.X[idx2].dot(self.X[idx2])
            if eta >= 0:
                continue
            alpha2_new_unc = self.alpha[idx2] - (y[idx2] * (error[idx1] - error[idx2])) / eta
            alpha2_new = min(max(alpha2_new_unc, 0), self.C)
            alpha1_new = self.alpha[idx1] + y[idx1] * y[idx2] * (self.alpha[idx2] - alpha2_new)
            self.alpha[idx1] = alpha1_new
            self.alpha[idx2] = alpha2_new
        self.calculate_bias()

    def partial_fit(self, X, y, classes=None):
        y = np.where(y == 0, -1, y)
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        # if self.alpha is None:
        #     self.alpha = np.zeros(X.shape[0])
        self.fit(X, y)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        # Ensure alpha has the same length as the number of samples
        if self.alpha.shape[0] != self.X.shape[0]:
            raise ValueError("Mismatch between alpha and X")

        y = np.sign(np.dot(self.alpha * self.y, self.X.dot(X.T)) + self.b)
        y = np.where(y == -1, 0, y)
        return y

    def select_two_instances(self, error):
        num_samples = len(error)
        positive_errors = np.where(error > self.tolerance)[0]
        negative_errors = np.where(error < -self.tolerance)[0]

        if len(positive_errors) > 0 and len(negative_errors) > 0:
            idx1 = positive_errors[np.argmax(error[positive_errors])]
            idx2 = negative_errors[np.argmax(-error[negative_errors])]
            return idx1, idx2
        else:
            return -1, -1

    def calculate_bias(self):
        support_vectors = np.where(self.alpha > 0)[0]
        bias_sum = 0
        for i in support_vectors:
            bias_sum += self.y[i] - np.dot(self.alpha * self.y, self.X.dot(self.X[i].T))
        self.b = bias_sum / len(support_vectors)

