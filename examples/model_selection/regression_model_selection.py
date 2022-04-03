from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from river import stream
from river import preprocessing, linear_model, optim
import time
import ray
from sail.model_selector.holdout_best_model import HoldoutBestModelSelector
from sklearn.metrics import r2_score
from sail.imla.ielm import IELM

ray.init()

boston = load_boston()
boston_data, boston_target = boston.data, boston.target

df_X = pd.DataFrame(boston_data, columns=boston.feature_names)
df_y = pd.Series(boston_target)

df_X = df_X.iloc[:10]
df_y = df_y.iloc[:10]

xtrain, xtest, ytrain, ytest = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

stdScaler_many = preprocessing.StandardScaler()

dataset = stream.iter_pandas(xtrain, ytrain)

x = xtrain
yi = ytrain

stdScaler_many.learn_many(x)
x = stdScaler_many.transform_many(x)

ielm_model = IELM(numInputNodes=13, numOutputNodes=1, numHiddenNodes=7)

offline_model = HoldoutBestModelSelector(estimators=[linear_model.LinearRegression(), ielm_model],
                                metrics=r2_score)

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


print(offline_model.get_best_model())
