import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from river import metrics, optim
from sklearn.impute import SimpleImputer

from sail.models.river.forest import (
    AdaptiveRandomForestClassifier,
    AdaptiveRandomForestRegressor,
)
from sail.models.river.linear_model import LinearRegression, LogisticRegression
from sail.pipeline import SAILPipeline
from sail.transformers.river.preprocessing import StandardScaler


@pytest.fixture(scope="module", autouse=True)
def seeds_fixed():
    np.random.seed(0)


@pytest.fixture(scope="module")
def regression_dataset(request):
    X = pd.read_csv(
        os.path.join(str(request.config.rootdir), "examples/datasets/HDWF2.csv")
    ).head(500)
    y = X["power"]
    X.drop(["power", "time"], axis=1, inplace=True)
    return X, y


@pytest.fixture(scope="module")
def classification_dataset(request):
    X = pd.read_csv(
        os.path.join(str(request.config.rootdir), "examples/datasets/agrawal.csv")
    ).head(500)
    y = X["class"]
    X.drop("class", axis=1, inplace=True)
    return X, y


@pytest.fixture(scope="module")
def classification_pipeline():
    steps = [
        ("Imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("standard_scalar", StandardScaler()),
        ("classifier", "passthrough"),
    ]
    return SAILPipeline(steps=steps, scoring=metrics.Accuracy)


@pytest.fixture(scope="module")
def regression_pipeline():
    steps = [
        ("Imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("standard_scalar", StandardScaler()),
        ("regressor", "passthrough"),
    ]
    return SAILPipeline(steps=steps, scoring=metrics.R2)


@pytest.fixture(scope="module")
def classification_models():
    logistic_reg = LogisticRegression(optimizer=optim.SGD(0.1))
    random_forest = AdaptiveRandomForestClassifier(n_models=10)
    return logistic_reg, random_forest


@pytest.fixture(scope="module")
def regression_models():
    linear_reg = LinearRegression(optimizer=optim.SGD(0.1))
    random_forest = AdaptiveRandomForestRegressor()
    return linear_reg, random_forest


@pytest.fixture(scope="module")
def create_tmp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)
