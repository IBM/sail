"""Tests for src/tsc.py.
Based on: https://github.com/skorch-dev/skorch/blob/master/skorch/tests/test_regressor.py
"""
import numpy as np
import pytest
import torch

class TestTSC:
    
    @pytest.fixture
    def data(self):
        classification_data = []
        from sklearn.datasets import make_classification
        X, y = make_classification(390, 176, n_informative=5, random_state=0)
        X, y = X.astype(np.float32), y.astype(np.int64)
        print(X.shape, y.shape)
        Y = torch.from_numpy(y).type(torch.LongTensor)
        classification_data.append(X)
        classification_data.append(y)
        return classification_data
        
    @pytest.fixture
    def net(self):
        from sail.models.torch.fcn_classification import ConvNet
        return ConvNet(n_in=176, n_classes=37)
    
    @pytest.fixture
    def net_fit(self, net, data):
        X, y = data
        net.fit(X, y)
        return net
    
    def test_predict_and_predict_proba(self, net_fit, data):
        X = data[0]
        y_proba = net_fit.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1, rtol=1e-5)

        y_pred = net_fit.predict(X)
        assert np.allclose(np.argmax(y_proba, 1), y_pred, rtol=1e-5)
        
    def test_score(self, net_fit, data):
        X, y = data
        r2_score = net_fit.score(X, y)
        assert r2_score <= 1. and r2_score > 0.1
    