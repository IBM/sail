import pytest
import os
import numpy as np
import torch
import torch.nn as nn


# methods for data generation and preprocessing
def load_to_torch(X_train, y_train, device):
    X_train = torch.from_numpy(X_train)
    X_train.requires_grad = False
    X_train = X_train.to(device)
    y_train = torch.from_numpy(y_train).to(device)

    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze_(1)
    return X_train, y_train


@pytest.fixture
def Xy_train():
    from sklearn.datasets import make_classification
    X, y = make_classification(30, 10, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.int64)
    
    device = "cpu"
    
    X, y = load_to_torch(X, y, device)
    return X, y


@pytest.fixture
def net(Xy_train):
    from sail.models.torch.os_cnn import OS_CNN_CLassifier
    # the model prints out the result every epoch
    X_train, y_train = Xy_train
    Max_kernel_size = 89
    start_kernel_size = 1
    input_channel = X_train.shape[1] # input channel size
    n_class = max(y_train) + 1 # output class number
    receptive_field_shape= min(int(X_train.shape[-1]/4),Max_kernel_size)
    return OS_CNN_CLassifier(n_class.item(), input_channel, receptive_field_shape, max_epochs=2)


@pytest.fixture
def net_partial_fit(net, Xy_train):
    X, y = Xy_train
    for i in range(0,3):
        net.partial_fit(X, y)
    return net


def test_net_learns(net, Xy_train):
    X, y = Xy_train
    for i in range(0,3):
        net.partial_fit(X, y)
    train_losses = net.history[:, 'train_loss']
    assert train_losses[0] > train_losses[-1]
    
    
def test_predict_predict_proba(net_partial_fit, Xy_train):
    X, _ = Xy_train
    y_pred = net_partial_fit.predict(X)
    assert not np.allclose(y_pred, 0)

    
def test_score(net_partial_fit, Xy_train):
    X, y = Xy_train
    r2_score = net_partial_fit.score(X, y)
    assert r2_score <= 1. and r2_score > 0.1