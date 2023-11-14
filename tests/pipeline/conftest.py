import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from river import metrics
from sail.models.river.forest import (
    AdaptiveRandomForestClassifier,
    AdaptiveRandomForestRegressor,
)
from sail.pipeline import SAILPipeline
from sail.transformers.river.preprocessing import StandardScaler


@pytest.fixture(scope="module", autouse=True)
def seeds_fixed():
    np.random.seed(0)


@pytest.fixture(scope="module")
def regression_dataset(request):
    X = pd.read_csv(
        os.path.join(str(request.config.rootdir), "examples/datasets/HDWF2.csv")
    ).head(5000)
    y = X["power"]
    X.drop(["power", "time"], axis=1, inplace=True)
    return X, y


@pytest.fixture(scope="module")
def classification_dataset(request):
    X = pd.read_csv(
        os.path.join(str(request.config.rootdir), "examples/datasets/agrawal.csv")
    ).head(5000)
    y = X["class"]
    X.drop("class", axis=1, inplace=True)
    return X, y


@pytest.fixture(scope="module")
def classification_pipeline():
    random_forest = AdaptiveRandomForestClassifier(n_models=10)
    steps = [
        ("Imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("standard_scalar", StandardScaler()),
        ("classifier", random_forest),
    ]
    return SAILPipeline(steps=steps, scoring=[metrics.Accuracy])


@pytest.fixture(scope="module")
def regression_pipeline():
    random_forest = AdaptiveRandomForestRegressor()
    steps = [
        ("Imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("standard_scalar", StandardScaler()),
        ("regressor", random_forest),
    ]
    return SAILPipeline(steps=steps, scoring=[metrics.Accuracy])


@pytest.fixture(scope="module")
def create_tmp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)
