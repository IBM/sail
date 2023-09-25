"""Tests for pytorch/rnn.py.
Based on: https://github.com/skorch-dev/skorch/blob/master/skorch/tests/test_regressor.py
"""

import numpy as np
import pytest


class TestRNN:
    @pytest.fixture
    def net(self):
        from sail.models.torch.rnn import RNNRegressor

        return RNNRegressor(
            input_units=10,
            output_units=1,
            hidden_units=20,
            n_hidden_layers=3,
            lr=0.01,
            cell_type="RNN",
        )

    @pytest.fixture
    def net_partial_fit(self, net, regression_data):
        X, y = regression_data
        for i in range(1, 50):
            net.partial_fit(X, y)
        return net

    def test_net_learns(self, net, regression_data):
        X, y = regression_data
        for i in range(0, 3):
            net.partial_fit(X, y)
        train_losses = net.history[:, "train_loss"]
        assert train_losses[0] > train_losses[-1]

    def test_predict_predict_proba(self, net_partial_fit, regression_data):
        X = regression_data[0]
        y_pred = net_partial_fit.predict(X)

        # predictions should not be all zeros
        assert not np.allclose(y_pred, 0)
        y_proba = net_partial_fit.predict_proba(X)

        # predict and predict_proba should be identical for regression
        assert np.allclose(y_pred, y_proba, atol=1e-6)

    def test_score(self, net_partial_fit, regression_data):
        X, y = regression_data
        r2_score = net_partial_fit.score(X, y)
        assert r2_score <= 1.0 and r2_score > 0.9
