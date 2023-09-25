import pytest
import numpy as np
import torch


# methods for data generation and preprocessing
def load_to_torch(X_train, y_train, device):
    X_train = torch.from_numpy(X_train)
    X_train.requires_grad = False
    X_train = X_train.to(device)
    y_train = torch.from_numpy(y_train).reshape((-1,)).to(device)

    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze_(1)
    return X_train, y_train


@pytest.fixture
def net(classification_data):
    from sail.models.torch.os_cnn import OS_CNN_CLassifier

    # the model prints out the result every epoch
    X_train, y_train = classification_data
    X_train, y_train = load_to_torch(X_train, y_train, "cpu")
    Max_kernel_size = 89
    start_kernel_size = 1
    input_channel = X_train.shape[1]  # input channel size
    n_class = max(y_train) + 1  # output class number
    receptive_field_shape = min(int(X_train.shape[-1] / 4), Max_kernel_size)
    return OS_CNN_CLassifier(n_class.item(), input_channel, receptive_field_shape)


@pytest.fixture
def net_partial_fit(net, classification_data):
    X, y = classification_data
    X, y = load_to_torch(X, y, "cpu")
    for i in range(0, 3):
        net.partial_fit(X, y)
    return net


def test_net_learns(net, classification_data):
    X, y = classification_data
    X, y = load_to_torch(X, y, "cpu")
    for i in range(0, 3):
        net.fit(X, y)
    train_losses = net.history[:, "train_loss"]
    assert train_losses[0] > train_losses[-1]


def test_predict_predict_proba(net_partial_fit, classification_data):
    X, y = classification_data
    X, y = load_to_torch(X, y, "cpu")
    y_pred = net_partial_fit.predict(X)
    assert not np.allclose(y_pred, 0)


def test_score(net_partial_fit, classification_data):
    X, y = classification_data
    X, y = load_to_torch(X, y, "cpu")
    r2_score = net_partial_fit.score(X, y)
    assert r2_score <= 1.0 and r2_score > 0.1
