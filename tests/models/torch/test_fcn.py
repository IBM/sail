# -*- coding: utf-8 -*-
"""
Based on: https://github.com/skorch-dev/skorch/blob/master/skorch/tests/test_classifier.py
"""


import numpy as np
import pytest
import torch

class TestFCN:
    @pytest.fixture
    def classifier(self, classification_data):
        from sail.models.torch.fcn import FCN_Classifier
        return FCN_Classifier(in_channels=1, input_size=10, lstm_layers = 8,classes=2)
    
    
    @pytest.fixture
    def classification_data(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(30, 10, n_informative=5, random_state=0)
        X, y = X.astype(np.float32), y.astype(np.int64)
        return X, y
    
    @pytest.fixture
    def classifier_partial_fit(self, classifier, classification_data):
        X, y = classification_data
        for i in range(0,10):
            classifier.partial_fit(X, y)
        return classifier
        
    @pytest.fixture
    def classifier_fit(self, classifier, classification_data):
        X, y = classification_data
        return classifier.fit(X, y)
        
    def test_net_learns(self, classifier, classification_data):
        X, y = classification_data
        classifier.fit(X, y)
        valid_acc = classifier.history[-1, 'valid_acc']
        assert valid_acc >= 0.5
        
    def test_net_learns_partial(self, classifier, classification_data):
        X, y = classification_data
        for i in range(0,10):
            classifier.partial_fit(X, y)
        train_losses = classifier.history[:, 'train_loss']
        assert train_losses[0] > train_losses[-1]
        valid_acc = classifier.history[-1, 'valid_acc']
        assert valid_acc >= 0.5
        
    def test_score(self, classifier_fit, classification_data):
        X, y = classification_data
        accuracy = classifier_fit.score(X, y)
        assert 0.5 <= accuracy <= 1.
        
    def test_score_partial(self, classifier_partial_fit, classification_data):
        X, y = classification_data
        accuracy = classifier_partial_fit.score(X, y)
        assert 0.5 <= accuracy <= 1.
     
    def test_predict_and_predict_proba(self, classifier_fit, classification_data):
        X = classification_data[0]
        y_proba = classifier_fit.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1, rtol=1e-5)
        y_pred = classifier_fit.predict(X)
        assert np.allclose(np.argmax(y_proba, 1), y_pred, rtol=1e-5)
        
    def test_predict_and_predict_proba_partial(self, classifier_partial_fit, classification_data):
        X = classification_data[0]
        y_proba = classifier_partial_fit.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1, rtol=1e-5)
        y_pred = classifier_partial_fit.predict(X)
        assert np.allclose(np.argmax(y_proba, 1), y_pred, rtol=1e-5)

    def test_history_default_keys(self, classifier_fit):
        expected_keys = {
            'train_loss', 'valid_loss', 'epoch', 'dur', 'batches', 'valid_acc'
        }
        for row in classifier_fit.history:
            assert expected_keys.issubset(row)
            
    def test_history_default_keys_partial(self, classifier_partial_fit):
        expected_keys = {
            'train_loss', 'valid_loss', 'epoch', 'dur', 'batches', 'valid_acc'
        }
        for row in classifier_partial_fit.history:
            assert expected_keys.issubset(row)


