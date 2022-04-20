""" Code mainly from: https://github.com/skorch-dev/skorch/blob/master/skorch/tests/conftest.py"""
"""Contains shared fixtures, hooks, etc."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


INFERENCE_METHODS = ['predict', 'predict_proba', 'forward', 'forward_iter']


###################
# shared fixtures #
###################


@pytest.fixture(autouse=True)
def seeds_fixed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope='module')
def regression_data():
    X, y = make_regression(
        1000, 10, n_informative=10, bias=0, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)
    Xt = StandardScaler().fit_transform(X)
    yt = StandardScaler().fit_transform(y)
    return Xt, yt


torch_installed = False
try:
    # pylint: disable=unused-import
    import torch

    torch_installed = True
except ImportError:
    pass

