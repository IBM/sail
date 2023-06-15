from tabnanny import verbose
import pytest
import numpy as np
from sail.models.river.linear_model import LogisticRegression


class TestRiverSerialization:
    def test_model_serialization(self, classification_dataset, create_tmp_dir):
        dirpath = create_tmp_dir
        X, y = classification_dataset
        X = np.array_split(X, 2)
        y = np.array_split(y, 2)

        # Initial training
        model_1 = LogisticRegression()
        model_1.partial_fit(X[0], y[0], list(set(y[0])))

        # record stats and save model
        model_1_weights = model_1.instance_.weights
        model_1_intercept = model_1.instance_.intercept
        model_1.save_model(dirpath)

        # load a new model and record stats
        model_2 = LogisticRegression()
        model_2 = model_2.load_model(dirpath)
        model_2_weights = model_2.instance_.weights
        model_2_intercept = model_2.instance_.intercept

        assert model_1_weights == model_2_weights
        assert model_1_intercept == model_2_intercept

        # second round of training
        model_2.partial_fit(X[1], y[1])
