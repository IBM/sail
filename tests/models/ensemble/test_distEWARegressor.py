# import tracemalloc
# from array import array

# import numpy as np
# from river import linear_model, optim
# from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator

# from sail.models.ensemble.distEWARegressor import DistEWARegressor

# tracemalloc.start()


# class TestDistEWARegressor:
#     def test_ewar(self, ray_setup):
#         stream = HyperplaneGenerator(random_state=1)

#         optimizers = [optim.SGD(0.01), optim.RMSProp(), optim.AdaGrad()]

#         # prepare the ensemble
#         learner = DistEWARegressor(
#             estimators=[
#                 linear_model.LinearRegression(optimizer=o, intercept_lr=0.1)
#                 for o in optimizers
#             ]
#         )
#         cnt = 0
#         max_samples = 50
#         y_pred = array("f")
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
#             learner.partial_fit(X, y)
#             cnt += 1
#         expected_predictions = np.array(
#             [
#                 0.5535104274749756,
#                 0.7640034556388855,
#                 0.07437397539615631,
#                 0.27591532468795776,
#             ]
#         )

#         assert np.allclose(y_pred, expected_predictions)
#         assert type(learner.predict(X)) == np.ndarray

#         top_stats = tracemalloc.take_snapshot().statistics("lineno")
#         print("[ Top 10 ]")
#         for stat in top_stats[:10]:
#             print(stat)

#         print("Action: ", ray_setup)
