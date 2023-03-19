import time

import numpy as np
import pandas as pd
import ray
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sail.model_selector.holdout_best_model import HoldoutBestModelSelector
from sail.models.river.linear_model import LogisticRegression
from sail.models.river.naive_bayes import BernoulliNB

ray.init()

scaler = StandardScaler()

sgd = LogisticRegression()
bnb = BernoulliNB()

offline_model = HoldoutBestModelSelector(
    estimators=[sgd, bnb], metrics=accuracy_score
)

# Ingestion
for index in range(2):
    csv_file = "/Users/dhaval/Projects/MORE/sail-version-bump/examples/datasets/agrawal_gen_2000.csv"
    X = pd.read_csv(
        csv_file,
        names=["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "y"],
    )

    y = X["y"].values
    X.drop("y", axis=1, inplace=True)
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    # y = y.reshape(len(y), 1)
    X = X.to_numpy()

    start = time.time()
    offline_model.partial_fit(X, y, np.unique(y))
    delta = time.time() - start
    print(f"({delta:.3f} seconds)")

print(offline_model.best_model_index)
print(offline_model.get_best_model())

ray.shutdown()
