import time

import numpy as np
import pandas as pd
from river import linear_model, optim, preprocessing, stream
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sail.model_selector.holdout_best_model import HoldoutBestModelSelector
from sail.models.native.ielm import IELM


class TestHoldoutBestRegressor:
    def test_hbr(self, ray_setup):
        boston = pd.read_csv(
            "http://lib.stat.cmu.edu/datasets/boston",
            sep="\\s+",
            skiprows=22,
            header=None,
        )
        boston_data, boston_target = (
            np.hstack([boston.values[::2, :], boston.values[1::2, :2]]),
            boston.values[1::2, 2],
        )

        df_X = pd.DataFrame(boston_data, columns=range(0, 13))
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
            estimators=[linear_model.LinearRegression(), ielm_model],
            metrics=r2_score,
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
            "best model ",
            (hedge.get_best_model()).river_estimator.__class__.__name__,
        )
        # print("best model ", hedge.get_best_model_index(x,yi))
        assert type(hedge.get_best_model()).__name__ == "River2SKLRegressor"
        assert hedge.get_best_model_index(x, yi) == 1
        # assert (hedge.get_best_model()).river_estimator == "LinearRegression"

        # assert np.allclose(y_pred, expected_predictions)
        # assert type(learner.predict(X)) == np.ndarray

        print("Action: ", ray_setup)
