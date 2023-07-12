from array import array
from river.datasets import synth
import numpy as np
from river import linear_model, optim

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
                linear_model.LinearRegression(optimizer=o, intercept_lr=0.1)
                for o in optimizers
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
                0.3045717477798462,
                1.1020143032073975,
                0.6729293465614319,
                0.8377850651741028,
            ]
        )

        assert np.allclose(y_pred, expected_predictions)
        assert type(learner.predict(X)) == np.ndarray

        learner = DistAggregateRegressor(
            estimators=[
                linear_model.LinearRegression(optimizer=o, intercept_lr=0.1)
                for o in optimizers
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
                1.0728034973144531,
                -0.14423903822898865,
                1.1754292249679565,
                0.2284281849861145,
            ]
        )
        assert np.allclose(y_pred, expected_predictions)
        assert type(learner.predict(X)) == np.ndarray

        print("Action: ", ray_setup)
