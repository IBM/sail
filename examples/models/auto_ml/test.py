import numpy as np
import pandas as pd
from river import optim
from river.drift import ADWIN
from ray.tune.search import BasicVariantGenerator
from sail.models.auto_ml.tune import SAILTuneGridSearchCV
from sail.models.river.forest import AdaptiveRandomForestClassifier
from sail.models.river.linear_model import LogisticRegression
from sail.models.auto_ml.auto_pipeline import SAILAutoPipeline
from sail.pipeline import SAILPipeline
from sklearn.impute import SimpleImputer
import ray.cloudpickle as cpickle
from sail.transfomers.river.preprocessing import StandardScaler

df = pd.read_csv("/Users/dhaval/Projects/SAIL/sail/examples/datasets/agrawal.csv").head(
    50000
)
X = df.copy()

y = X["class"]
X.drop("class", axis=1, inplace=True)

logistic_reg = LogisticRegression(optimizer=optim.SGD(0.1))
random_forest = AdaptiveRandomForestClassifier(n_models=10)

steps = [
    ("Imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
    ("standard_scalar", StandardScaler()),
    ("classifier", logistic_reg),
]
sail_pipeline = SAILPipeline(steps=steps)

params_grid = [
    {
        "classifier": [logistic_reg],
        "classifier__l2": [0.1, 0.9],
        "classifier__intercept_init": [0.2, 0.5],
    },
    {
        "classifier": [random_forest],
        "classifier__n_models": [5, 10],
        "Imputer": ["passthrough"],
    },
]

auto_pipeline = SAILAutoPipeline(
    pipeline=sail_pipeline,
    pipeline_params_grid=params_grid,
    search_data_size=1000,
    search_method=SAILTuneGridSearchCV,
    search_method_params={
        "max_iters": 1,
        "early_stopping": False,
        "mode": "max",
        "scoring": "accuracy",
        "pipeline_auto_early_stop": False,
    },
    drift_detector=ADWIN(delta=0.001),
    pipeline_strategy="DetectAndWarmStart",
    keep_best_configurations=2,
)

y_preds = []
y_true = []
batch_size = 50

start = 0
for end in range(50, 1001, batch_size):
    X_train = X.iloc[start:end]
    y_train = y.iloc[start:end]

    auto_pipeline.train(X_train, y_train)
    start = end
