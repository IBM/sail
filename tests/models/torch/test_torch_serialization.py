import pytest
import numpy as np
from sail.models.torch import RNNRegressor


class TestTorchSerialization:
    def rnn_regressor(self):
        model = RNNRegressor(
            input_units=10,
            output_units=1,
            hidden_units=20,
            n_hidden_layers=3,
            lr=0.01,
            cell_type="RNN",
        )

        return model

    def test_model_serialization(self, regression_data, create_tmp_dir):
        dirpath = create_tmp_dir
        X, y = regression_data
        X = np.array_split(X, 2)
        y = np.array_split(y, 2)

        # Initial training
        model_1 = self.rnn_regressor()
        model_1.partial_fit(X[0], y[0])

        # record stats and save model
        model_1_weights = {}
        for name, param in model_1.module_.named_parameters():
            model_1_weights[name] = param.data.numpy()
        model_1_loss = model_1.history[:, "train_loss"]
        model_1.save_model(dirpath)

        # load a new model and record stats
        model_2 = RNNRegressor.load_model(dirpath)
        model_2_weights = {}
        for name, param in model_2.module_.named_parameters():
            model_2_weights[name] = param.data.numpy()
        model_2_loss = model_2.history[:, "train_loss"]

        for name, weights_1 in model_1_weights.items():
            weights_2 = model_2_weights[name]
            assert np.array_equal(weights_1, weights_2, equal_nan=True)
        assert len(set(model_1_loss).symmetric_difference(set(model_2_loss))) == 0

        # second round of training
        model_2.partial_fit(X[1], y[1])
