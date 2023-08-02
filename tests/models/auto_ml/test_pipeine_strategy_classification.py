import numpy as np
from sail.models.auto_ml.auto_pipeline import SAILAutoPipeline
from sail.models.auto_ml.tune import SAILTuneGridSearchCV
from sail.drift_detection.drift_detector import SAILDriftDetector
from river.drift.binary import EDDM


class TestPipelineStrategy:
    def get_params_grid(self, classification_models):
        logistic_reg, random_forest = classification_models
        return [
            {
                "classifier": [logistic_reg],
                "classifier__l2": [0.1],
                "classifier__intercept_init": [0.2, 0.5],
            },
            {
                "classifier": [random_forest],
                "classifier__n_models": [5, 10],
                "Imputer": ["passthrough"],
            },
        ]

    def test_detect_and_increment(
        self, classification_pipeline, classification_models, classification_dataset
    ):
        X, y = classification_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=classification_pipeline,
            pipeline_params_grid=self.get_params_grid(classification_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "accuracy",
                "pipeline_auto_early_stop": False,
                "keep_best_configurations": 2,
            },
            search_data_size=100,
            incremental_training=True,
            drift_detector=SAILDriftDetector(model=EDDM(), drift_param="difference"),
            pipeline_strategy="DetectAndIncrement",
        )

        batch_size = 50
        for start in range(0, 201, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            auto_pipeline.train(X_train, y_train, classifier__classes=[1, 0])

    def test_detect_and_retrain(
        self, classification_pipeline, classification_models, classification_dataset
    ):
        X, y = classification_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=classification_pipeline,
            pipeline_params_grid=self.get_params_grid(classification_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "accuracy",
                "pipeline_auto_early_stop": False,
                "keep_best_configurations": 2,
            },
            search_data_size=100,
            incremental_training=True,
            drift_detector=SAILDriftDetector(model=EDDM(), drift_param="difference"),
            pipeline_strategy="DetectAndRetrain",
        )

        batch_size = 50
        for start in range(0, 201, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            auto_pipeline.train(X_train, y_train, classifier__classes=[1, 0])

    def test_detect_and_warm_start(
        self, classification_pipeline, classification_models, classification_dataset
    ):
        X, y = classification_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=classification_pipeline,
            pipeline_params_grid=self.get_params_grid(classification_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "accuracy",
                "pipeline_auto_early_stop": False,
                "keep_best_configurations": 2,
            },
            search_data_size=100,
            incremental_training=True,
            drift_detector=SAILDriftDetector(model=EDDM(), drift_param="difference"),
            pipeline_strategy="DetectAndWarmStart",
        )

        batch_size = 50
        for start in range(0, 201, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            auto_pipeline.train(X_train, y_train, classifier__classes=[1, 0])

    def test_detect_and_restart(
        self, classification_pipeline, classification_models, classification_dataset
    ):
        X, y = classification_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=classification_pipeline,
            pipeline_params_grid=self.get_params_grid(classification_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "accuracy",
                "pipeline_auto_early_stop": False,
                "keep_best_configurations": 2,
            },
            search_data_size=100,
            incremental_training=True,
            drift_detector=SAILDriftDetector(model=EDDM(), drift_param="difference"),
            pipeline_strategy="DetectAndRestart",
        )

        batch_size = 50
        for start in range(0, 201, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            auto_pipeline.train(X_train, y_train, classifier__classes=[1, 0])

    def test_detect_and_periodic_restart(
        self, classification_pipeline, classification_models, classification_dataset
    ):
        X, y = classification_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=classification_pipeline,
            pipeline_params_grid=self.get_params_grid(classification_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "accuracy",
                "pipeline_auto_early_stop": False,
                "keep_best_configurations": 2,
            },
            search_data_size=100,
            incremental_training=True,
            drift_detector=SAILDriftDetector(model=EDDM(), drift_param="difference"),
            pipeline_strategy="PeriodicRestart",
        )

        batch_size = 50
        for start in range(0, 201, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            auto_pipeline.train(X_train, y_train, classifier__classes=[1, 0])

    def test_prequential_training(
        self, classification_pipeline, classification_models, classification_dataset
    ):
        X, y = classification_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=classification_pipeline,
            pipeline_params_grid=self.get_params_grid(classification_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "accuracy",
                "pipeline_auto_early_stop": False,
                "keep_best_configurations": 2,
            },
            search_data_size=100,
            pipeline_strategy="PrequentialTraining",
        )

        batch_size = 50
        for start in range(0, 201, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            auto_pipeline.train(X_train, y_train, classifier__classes=[1, 0])
