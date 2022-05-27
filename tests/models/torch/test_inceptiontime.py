import pytest

'''
UNIT TESTING
'''

class TestInceptionTime:
    @pytest.fixture
    def net(self):
        from  sail.models.torch.rnn.inceptiontime import InceptionTimeClassifier
        return InceptionTimeClassifier(lr=0.07)

    @pytest.fixture
    def net_partial_fit(self, net, classification_x, classification_y):
        X = classification_x
        y = classification_y
        for i in range(1,3):
            net.partial_fit(X, y)
        return net

    def test_net_learns(self, net, classification_x, classification_y):
        X = classification_x
        y = classification_y
        for i in range(0,3):
            net.partial_fit(X, y)
        train_losses = net.history[:, 'train_loss']
        assert train_losses[0] > train_losses[-1]

    def test_predict(self, net_partial_fit, classification_x, classification_y):
        X = classification_x
        y_pred = net_partial_fit.predict(X)

        # predictions should not be all zeros
        assert not np.allclose(y_pred, 0)
