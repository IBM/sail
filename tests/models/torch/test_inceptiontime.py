import numpy as np
import torch
import pytest

'''
UNIT TESTING
'''

class TestInceptionTime:

    @pytest.fixture
    def classification_dataset(self):
        def readucr(filename):
            data = np.loadtxt(filename, delimiter="\t")
            y = data[:, 0]
            x = data[:, 1:]
            return x, y.astype(int)


        root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

        x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
        x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], ))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        y_train=y_train.unsqueeze(1)

        x_train=x_train.to(torch.float32)
        y_train=y_train.to(torch.float32)
        return x_train, y_train

    @pytest.fixture
    def net(self):
        from  sail.models.torch.inceptiontime import InceptionTimeClassifier
        return InceptionTimeClassifier(out_channels=10, bottleneck_channels=10, batch_size=500, max_epochs=20, learning_rate=0.5, device='cpu')

    @pytest.fixture
    def net_partial_fit(self, net, classification_dataset):
        X, y = classification_dataset
        net.partial_fit(X[0:10],y[0:10])
        return net

    def test_net_learns(self, net, net_partial_fit):
        train_losses = net.history[:, 'train_loss']
        assert train_losses[0] > train_losses[-1]

    def test_predict(self, net_partial_fit, classification_dataset):
        X, y = classification_dataset
        y_pred = net_partial_fit.predict(X[20:30])

        # predictions should not be all zeros
        assert not np.allclose(y_pred, 0)
