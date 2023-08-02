import tracemalloc
from array import array
from river.datasets import synth
import numpy as np
from river import linear_model, optim


from sail.models.ensemble.distEWARegressor import DistEWARegressor

tracemalloc.start()


def hyperplane_stream(random_state, n_features):
    stream = synth.Hyperplane(seed=random_state, n_features=n_features)
    for x, y in stream.take(100):
        yield np.array(list(x.values())).reshape((1, n_features)), np.array(y).reshape(
            (1,)
        )


class TestDistEWARegressor:
    def test_ewar(self, ray_setup):
        stream = hyperplane_stream(random_state=1, n_features=10)

        optimizers = [optim.SGD(0.01), optim.RMSProp(), optim.AdaGrad()]

        # prepare the ensemble
        learner = DistEWARegressor(
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
                0.17334376275539398,
                0.22032581269741058,
                0.3394103944301605,
                0.6385474801063538,
            ]
        )
        assert np.allclose(y_pred, expected_predictions)
        assert type(learner.predict(X)) == np.ndarray

        top_stats = tracemalloc.take_snapshot().statistics("lineno")
        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)

        print("Action: ", ray_setup)
