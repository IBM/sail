"""Tests for pytorch/rnn.py.
Based on: https://github.com/skorch-dev/skorch/blob/master/skorch/tests/test_regressor.py
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler


class TestTCN:
    @pytest.fixture
    def net(self):
        from sail.models.torch.tcn import TCNRegressor

        return TCNRegressor(input_dim=10, output_dim=1)

    @pytest.fixture
    def net_partial_fit(self, net, regression_data):
        X, y = regression_data
        X = X[:, :, np.newaxis]
        y = StandardScaler().fit_transform(y.reshape((-1, 1)))
        for _ in range(50):
            net.partial_fit(X, y)
        return net

    def test_net_learns(self, net, regression_data):
        X, y = regression_data
        X = X[:, :, np.newaxis]
        y = StandardScaler().fit_transform(y.reshape((-1, 1)))
        for _ in range(3):
            net.partial_fit(X, y)
        train_losses = net.history[:, "train_loss"]
        assert train_losses[0] > train_losses[-1]

    def test_predict_predict_proba(self, net_partial_fit, regression_data):
        X = regression_data[0]
        X = X[:, :, np.newaxis]

        y_pred = net_partial_fit.predict(X)

        # predictions should not be all zeros
        assert not np.allclose(y_pred, 0)
        y_proba = net_partial_fit.predict_proba(X)

        # predict and predict_proba should be identical for regression
        assert np.allclose(y_pred, y_proba, atol=1e-6)

    def test_score(self, net_partial_fit, regression_data):
        X, y = regression_data
        X = X[:, :, np.newaxis]
        y = StandardScaler().fit_transform(y.reshape((-1, 1)))
        r2_score = net_partial_fit.score(X, y)
        assert r2_score <= 1.0 and r2_score > 0.9
