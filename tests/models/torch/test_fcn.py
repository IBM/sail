"""Tests for src/tsc.py.
Based on: https://github.com/skorch-dev/skorch/blob/master/skorch/tests/test_regressor.py
"""
import numpy as np
import pytest
import torch
from src import utils
from sklearn.datasets import make_classification

from skorch.tests.conftest import INFERENCE_METHODS

class TestTSC:
    
    @pytest.fixture
    def data(self):
        classification_data = []
        datasets = ['Adiac']
   
        dataset_dictionary = utils.data_dictionary(datasets)
        for dataset, dataloader in dataset_dictionary.items():
            # For using the downloaded dataset, a dataset and a dataloader are created. 
            # Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples
            x_train = dataloader['train'].dataset.x
            x_train = np.array(x_train)
            # x_train = torch.from_numpy(x_train).type(torch.LongTensor)
            
            y_train = dataloader['train'].dataset.y
            y_train = np.array(y_train)
            y_train = torch.from_numpy(y_train).type(torch.LongTensor)
            
            classification_data.append(x_train)
            classification_data.append(y_train)
        return classification_data
    
    @pytest.fixture
    def net(self):
        from src import ConvNet
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
    