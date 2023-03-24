import time

import numpy as np
import pandas as pd
import ray
from river import stream
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sail.model_selector.holdout_best_model import HoldoutBestModelSelector
from sail.models.native.ielm import IELM
from sail.models.river.linear_model import LinearRegression
from sail.models.river.preprocessing import StandardScaler

ray.init()

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

xtrain, xtest, ytrain, ytest = train_test_split(
    df_X, df_y, test_size=0.2, random_state=42
)

stdScaler_many = StandardScaler()

dataset = stream.iter_pandas(xtrain, ytrain)

x = xtrain
yi = ytrain

stdScaler_many.partial_fit(x)
x = stdScaler_many.transform(x)

ielm_model = IELM(numInputNodes=13, numOutputNodes=1, numHiddenNodes=7)

offline_model = HoldoutBestModelSelector(
    estimators=[LinearRegression(), ielm_model], metrics=r2_score
)

start = time.time()
offline_model.partial_fit(x, yi)
print("duration =", time.time() - start, "\nresult = ", sum)

start = time.time()
offline_model.partial_fit(x, yi)
print("duration =", time.time() - start, "\nresult = ", sum)

start = time.time()
offline_model.partial_fit(x, yi)
print("duration =", time.time() - start, "\nresult = ", sum)

start = time.time()
offline_model.partial_fit(x, yi)
print("duration =", time.time() - start, "\nresult = ", sum)


print("Best Model: ", offline_model.get_best_model())

ray.shutdown()
