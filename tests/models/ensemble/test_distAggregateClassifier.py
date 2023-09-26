from array import array

import numpy as np
from river import optim
from sail.models.river.linear_model import LogisticRegression
from river.datasets import synth

from sail.models.ensemble.distAggregateClassifier import DistAggregateClassifier


def classification_stream():
    stream = synth.STAGGER(seed=1, balance_classes=True)
    for x, y in stream.take(100):
        yield np.array(list(x.values())).reshape((1, 3)), np.array(y).reshape((1,))


class TestDistAggregateClassifier:
    def test_dac(self, ray_setup):
        # stream = SEAGenerator(random_state=1)
        stream = classification_stream()
        optimizers = [optim.SGD(0.01), optim.RMSProp(), optim.AdaGrad()]

        # prepare the ensemble
        learner = DistAggregateClassifier(
            estimators=[
                LogisticRegression(optimizer=o, intercept_lr=0.1) for o in optimizers
            ]
        )
        cnt = 0
        max_samples = 50
        y_pred = array("i")
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
            learner.partial_fit(X, y, classes=[0, 1])
            cnt += 1

        expected_predictions = array("i", [0, 1, 1, 1])
        assert np.all(y_pred == expected_predictions)
        assert type(learner.predict(X)) == np.ndarray

        learner = DistAggregateClassifier(
            estimators=[
                LogisticRegression(optimizer=o, intercept_lr=0.1) for o in optimizers
            ],
            aggregator="majority_vote",
        )
        cnt = 0
        max_samples = 50
        y_pred = array("i")
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
            learner.partial_fit(X, y, classes=[0, 1])
            cnt += 1

        expected_predictions = array("i", [0, 0, 1, 0])
        assert np.all(y_pred == expected_predictions)
        assert type(learner.predict(X)) == np.ndarray

        print("Action: ", ray_setup)
