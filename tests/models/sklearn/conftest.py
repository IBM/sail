import numpy as np
import pytest
import tempfile
import shutil
from sklearn import datasets
from sail.models.sklearn.preprocessing import StandardScaler


@pytest.fixture(autouse=True)
def seeds_fixed():
    np.random.seed(0)


@pytest.fixture(scope="module")
def classification_dataset():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.fixture(scope="module")
def create_tmp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)
