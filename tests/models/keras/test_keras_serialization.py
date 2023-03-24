import numpy as np

from sail.models.keras import WGLSTM
from sail.utils.ts_utils import generate_features_and_targets


class TestKerasSerialization:
    def wglstm_model(self):
        num_of_features = 1
        timesteps = 1
        window_size = 20

        model = WGLSTM(
            loss="mse",
            epochs=1,
            verbose=1,
            num_of_features=num_of_features,
            hidden_layer_neurons=450,
            hidden_layer_activation="linear",
            regularization_factor=0.0001,
            timesteps=timesteps,
            window_size=window_size,
        )

        return model

    def test_model_serialization(self, regression_dataset, create_tmp_dir):
        dirpath = create_tmp_dir
        X, y = generate_features_and_targets(
            regression_dataset, "Global_active_power"
        )
        model_1 = self.wglstm_model()

        # Initial training
        model_1.partial_fit(X, y)

        # record stats and save model
        model_1_weights = model_1.model_.get_weights()
        model_1_loss = model_1.model_.losses[0].numpy()
        model_1.save(dirpath)

        # load a new model and record stats
        model_2 = self.wglstm_model()
        model_2.load(dirpath)
        model_2.initialize(X, y)

        model_2_weights = model_2.model_.get_weights()
        for weights_1, weights_2 in zip(model_1_weights, model_2_weights):
            assert np.array_equal(weights_1, weights_2, equal_nan=True)

        model_2_loss = model_2.model_.losses[0].numpy()
        assert model_1_loss == model_2_loss

        # second round of training
        model_2.partial_fit(X, y)
