"""
In this example we're going to forecast the number of bikes in 5 bike stations from the city of Toulouse.
We'll do so by using sail's river wrapper.
This tutorial is based on rivers' own example: https://riverml.xyz/dev/examples/bike-sharing-forecasting/.
Different from river's tutorial, we will avoid using the function evaluate.progressive_val_score.
We will do so by controlling our self the training and evaluating loop.
"""

import numpy as np
import pandas as pd
from river import datasets
from river import metrics
from river import stats
from river import optim
from sail.models.river.feature_extraction import TargetAgg
from sail.models.river.linear_model import LinearRegression
from sail.models.river.preprocessing import StandardScaler


# Loading the dataset
dataset = datasets.Bikes()
x, y = [], []
for data, label in dataset:
    x.append(data)
    y.append(label)

df = pd.DataFrame(x)
# These are the features that we are interested in
keep_col = ['clouds', 'humidity', 'pressure', 'temperature', 'wind']
X = df[keep_col].copy()

metric = metrics.MAE()

# A simple linear regression will do the job
model = LinearRegression(optimizer=optim.SGD(0.001))

# Now we create the StandarScaler.
# However, we need the river estimator to call the internal procedures of river's StandarScaler
scaler = StandardScaler().river_estimator

# A for loop over the elements of the dataset
for i in range(X.shape[0]):
    x = X.iloc[i]

    # Here we apply the incremental StandarScaler to update the values and transform the new example
    x = np.asarray(list(scaler.learn_one(x).transform_one(x).values())).reshape(1, -1)

    # Partial_fit to update the linear regression's parameters
    model = model.partial_fit(x, [y[i]])

    # Predicting
    yhat = model.predict(x)

    # Update the metric
    metric.update([y[i]], yhat)
    if i % 10000 == 0:
        print('MAE after', i, 'iterations', metric.get())

print('Finally, MAE:', metric.get())

#  We can improve the results by taking the average number of bikes per hour for each station as a new feature

#  Adding the station and moment to the dataset
X = df[keep_col + ['station', 'moment']].copy()

metric = metrics.MAE()

model = LinearRegression(optimizer=optim.SGD(0.001))

# This method allows to apply a transformation to the target variable based on a set features.
# In this case, the transformation we are looking for is to compute the average
agg = TargetAgg(by=['station', 'hour'], how=stats.BayesianMean(prior=1,
                                                               prior_weight=1)).river_estimator
scaler = StandardScaler().river_estimator

for i in range(X.shape[0]):
    x = X.iloc[i]
    #  Getting the hour
    x = x.append(pd.Series({'hour': x.moment.hour}))

    #  Getting the current average number of bikes per each hour and station
    agg_feature = agg.transform_one(x)
    x = x.append(pd.Series(agg_feature))

    #  Removing the extra features
    input_x = x.drop(['station', 'hour', 'moment'])

    #  Scaling up the current example
    input_x = np.asarray(list(scaler.learn_one(input_x).transform_one(input_x).values())).reshape(1, -1)

    # Partially fitting the model
    model = model.partial_fit(input_x, [y[i]])

    yhat = model.predict(input_x)

    metric.update([y[i]], yhat)
    if i % 10000 == 0:
        print('MAE after', i, 'iterations', metric.get())

    agg = agg.learn_one(x, y[i])  # Updating the average number of bikes

print('Finally, MAE:', metric.get())

# The model have improved considerably by adding the average number of bikes.
# However, in real life scenarios we will not be able to update the average number of bikes immediately.
# Instead, we will have to wait for some time before having that true values.
# River's evaluate.progressive_val_score allows you to simulate this real life scenarios by adding a "delay".
# For more information: https://riverml.xyz/dev/api/stream/simulate-qa/
