import numpy as np
import pytest
import tempfile
import shutil
import pandas as pd
from sklearn import datasets
from sail.models.river.preprocessing import StandardScaler


@pytest.fixture(scope="module", autouse=True)
def seeds_fixed():
    np.random.seed(0)


@pytest.fixture(scope="module")
def regression_dataset():
    df = pd.read_csv(
        "examples/datasets/household_power_consumption.csv",
        sep=",",
        low_memory=False,
    )[600:700]
    return df


@pytest.fixture(scope="module")
def create_tmp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)
