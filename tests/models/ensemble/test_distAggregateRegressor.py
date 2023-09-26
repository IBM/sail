from array import array
from river.datasets import synth
import numpy as np
from river import optim
from sail.models.river.linear_model import LinearRegression

from sail.models.ensemble.distAggregateRegressor import DistAggregateRegressor


def hyperplane_stream(n_features):
    stream = synth.Hyperplane(seed=42, n_features=n_features)
    for x, y in stream.take(100):
        yield np.array(list(x.values())).reshape((1, n_features)), np.array(y).reshape(
            (1,)
        )


class TestDistAggregateRegressor:
    def test_dar(self, ray_setup):
        stream = hyperplane_stream(n_features=10)  # HyperplaneGenerator(random_state=1)
        optimizers = [optim.SGD(0.01), optim.RMSProp(), optim.AdaGrad()]

        # prepare the ensemble
        learner = DistAggregateRegressor(
            estimators=[
                LinearRegression(optimizer=o, intercept_lr=0.1) for o in optimizers
            ]
        )
        cnt = 0
        max_samples = 50
        y_pred = array("f")
        X_batch = []
        y_batch = []
        wait_samples = 10

        while cnt < max_samples:
            X, y = next(stream)
            X_batch.append(X[0])
            y_batch.append(y[0])
            # Test every n samples
            if (cnt % wait_samples == 0) and (cnt != 0):
                y_pred.append(learner.predict(X)[0])
            learner.partial_fit(X, y)
            cnt += 1
        expected_predictions = np.array(
            [
                0.44149839878082275,
                0.4521682560443878,
                0.35752978920936584,
                0.8726186752319336,
            ]
        )

        assert np.allclose(y_pred, expected_predictions)
        assert type(learner.predict(X)) == np.ndarray

        learner = DistAggregateRegressor(
            estimators=[
                LinearRegression(optimizer=o, intercept_lr=0.1) for o in optimizers
            ],
            aggregator="windsor",
        )
        cnt = 0
        max_samples = 50
        y_pred = array("f")
        X_batch = []
        y_batch = []
        wait_samples = 10

        while cnt < max_samples:
            X, y = next(stream)
            X_batch.append(X[0])
            y_batch.append(y[0])
            # Test every n samples
            if (cnt % wait_samples == 0) and (cnt != 0):
                y_pred.append(learner.predict(X)[0])
            learner.partial_fit(X, y)
            cnt += 1
        expected_predictions = np.array(
            [
                0.7171313762664795,
                0.8493690490722656,
                0.6237673759460449,
                -0.03409786522388458,
            ]
        )
        assert np.allclose(y_pred, expected_predictions)
        assert type(learner.predict(X)) == np.ndarray

        print("Action: ", ray_setup)
