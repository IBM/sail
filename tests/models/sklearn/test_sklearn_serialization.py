from tabnanny import verbose
import pytest
import numpy as np
from sail.models.sklearn.linear_model import SGDClassifier
from sail.models.sklearn.base import save, load


class TestSkleanSerialization:
    def test_model_serialization(self, classification_dataset, create_tmp_dir):
        dirpath = create_tmp_dir
        X, y = classification_dataset
        X = np.array_split(X, 2)
        y = np.array_split(y, 2)

        # Initial training
        model_1 = SGDClassifier(tol=1e-3)
        a = model_1.partial_fit(X[0], y[0], list(set(y[0])))

        # record stats and save model
        model_1_coef = model_1.coef_
        model_1_intercept = model_1.intercept_
        save(model_1, dirpath)

        # load a new model and record stats
        model_2 = SGDClassifier(tol=1e-3)
        model_2 = load(dirpath)
        model_2_coef = model_2.coef_
        model_2_intercept = model_2.intercept_

        assert np.array_equal(model_1_coef, model_2_coef, equal_nan=True)
        assert np.array_equal(model_1_intercept, model_2_intercept, equal_nan=True)

        # second round of training
        model_2.partial_fit(X[1], y[1])
