import unittest
import warnings
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from river import stream
from river import preprocessing, linear_model, optim
import time
import ray
from sail.model_selector.holdout_best_model import HoldoutBestModelSelector
from sklearn.metrics import r2_score
from sail.models.native.ielm import IELM


class TestHoldoutBestRegressor(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    def tearDown(self):
        warnings.simplefilter("default", ResourceWarning)

    @classmethod
    def setUpClass(cls):
        ray.init(local_mode=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_hbr(self):
        boston = load_boston()
        boston_data, boston_target = boston.data, boston.target

        df_X = pd.DataFrame(boston_data, columns=boston.feature_names)
        df_y = pd.Series(boston_target)

        df_X = df_X.iloc[:10]
        df_y = df_y.iloc[:10]
        # print(df_X.shape)

        xtrain, xtest, ytrain, ytest = train_test_split(
            df_X, df_y, test_size=0.2, random_state=42
        )

        stdScaler_many = preprocessing.StandardScaler()

        model_many = linear_model.LinearRegression(optimizer=optim.RMSProp())

        dataset = stream.iter_pandas(xtrain, ytrain)

        x = xtrain
        yi = ytrain

        stdScaler_many.learn_many(x)
        x = stdScaler_many.transform_many(x)

        ielm_model = IELM(numInputNodes=13, numOutputNodes=1, numHiddenNodes=7)

        hedge = HoldoutBestModelSelector(
            estimators=[linear_model.LinearRegression(), ielm_model], metrics=r2_score
        )

        start = time.time()
        hedge.partial_fit(x, yi)
        print("duration =", time.time() - start, "\nresult = ", sum)

        start = time.time()
        hedge.partial_fit(x, yi)
        print("duration =", time.time() - start, "\nresult = ", sum)

        hedge.predict(x)

        # print("best model ", type(hedge.get_best_model()).__name__)
        print(
            "best model ", (hedge.get_best_model()).river_estimator.__class__.__name__
        )
        # print("best model ", hedge.get_best_model_index(x,yi))
        assert type(hedge.get_best_model()).__name__ == "River2SKLRegressor"
        assert hedge.get_best_model_index(x, yi) == 1
        # assert (hedge.get_best_model()).river_estimator == "LinearRegression"

        # assert np.allclose(y_pred, expected_predictions)
        # assert type(learner.predict(X)) == np.ndarray


if __name__ == "__main__":
    unittest.main()
