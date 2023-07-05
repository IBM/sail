# from array import array

# import numpy as np
# from river import linear_model, optim
# from skmultiflow.data import SEAGenerator

# from sail.models.ensemble.distAggregateClassifier import (
#     DistAggregateClassifier,
# )


# class TestDistAggregateClassifier:
#     __test__ = False

#     def test_dac(self, ray_setup):
#         stream = SEAGenerator(random_state=1)

#         optimizers = [optim.SGD(0.01), optim.RMSProp(), optim.AdaGrad()]

#         # prepare the ensemble
#         learner = DistAggregateClassifier(
#             estimators=[
#                 linear_model.LogisticRegression(optimizer=o, intercept_lr=0.1)
#                 for o in optimizers
#             ]
#         )
#         cnt = 0
#         max_samples = 50
#         y_pred = array("i")
#         X_batch = []
#         y_batch = []
#         wait_samples = 10

#         while cnt < max_samples:
#             X, y = stream.next_sample()
#             X_batch.append(X[0])
#             y_batch.append(y[0])
#             # Test every n samples
#             if (cnt % wait_samples == 0) and (cnt != 0):
#                 y_pred.append(learner.predict(X)[0])
#             learner.partial_fit(X, y, classes=stream.target_values)
#             cnt += 1

#         expected_predictions = array("i", [1, 1, 1, 0])
#         assert np.all(y_pred == expected_predictions)
#         assert type(learner.predict(X)) == np.ndarray

#         learner = DistAggregateClassifier(
#             estimators=[
#                 linear_model.LogisticRegression(optimizer=o, intercept_lr=0.1)
#                 for o in optimizers
#             ],
#             aggregator="majority_vote",
#         )
#         cnt = 0
#         max_samples = 50
#         y_pred = array("i")
#         X_batch = []
#         y_batch = []
#         wait_samples = 10

#         while cnt < max_samples:
#             X, y = stream.next_sample()
#             X_batch.append(X[0])
#             y_batch.append(y[0])
#             # Test every n samples
#             if (cnt % wait_samples == 0) and (cnt != 0):
#                 y_pred.append(learner.predict(X)[0])
#             learner.partial_fit(X, y, classes=stream.target_values)
#             cnt += 1

#         expected_predictions = array("i", [1, 1, 1, 1])
#         assert np.all(y_pred == expected_predictions)
#         assert type(learner.predict(X)) == np.ndarray

#         print("Action: ", ray_setup)
