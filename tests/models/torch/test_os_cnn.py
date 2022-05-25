import pytest
import os
from os.path import dirname
# os.chdir("..")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing


from sail.models.torch.os_cnn import OS_CNN_CLassifier

# methods for preprocessing data 
def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def TSC_data_loader(dataset_path,dataset_name):
    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test


def load_to_torch(X_train, y_train, X_test, y_test, device):
    X_train = torch.from_numpy(X_train)
    X_train.requires_grad = False
    X_train = X_train.to(device)
    y_train = torch.from_numpy(y_train).to(device)

    X_test = torch.from_numpy(X_test)
    X_test.requires_grad = False
    X_test = X_test.to(device)
    y_test = torch.from_numpy(y_test).to(device)


    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze_(1)
        X_test = X_test.unsqueeze_(1)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def Xy_train():
    dataset_path = dirname("./notebooks/OS_CNN/UCRArchive_2018/")
    dataset_name = "FiftyWords"
    device = "cpu"
    # load data,
    X_train, y_train, X_test, y_test = TSC_data_loader(dataset_path, dataset_name)
    
    X_train, y_train, X_test, y_test = load_to_torch(X_train, y_train, X_test, y_test, device)
    return X_train, y_train


@pytest.fixture
def net(Xy_train):
    # the model prints out the result every epoch
    # defaul epoch size = 20
    X_train, y_train = Xy_train
    Max_kernel_size = 89
    start_kernel_size = 1
    input_channel = X_train.shape[1] # input channel size
    n_class = max(y_train) + 1 # output class number
    receptive_field_shape= min(int(X_train.shape[-1]/4),Max_kernel_size)
    return OS_CNN_CLassifier(n_class.item(), input_channel, receptive_field_shape)


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
    assert r2_score <= 1. and r2_score > 0.5