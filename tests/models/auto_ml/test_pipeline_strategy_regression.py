import numpy as np
from sail.models.auto_ml.auto_pipeline import SAILAutoPipeline
from sail.models.auto_ml.tune import SAILTuneGridSearchCV
from sail.drift_detection.drift_detector import SAILDriftDetector
from river.drift.binary import EDDM


class TestPipelineStrategy:
    def get_params_grid(self, regression_models):
        linear_reg, random_forest = regression_models
        return [
            {
                "regressor": [linear_reg],
                "regressor__l2": [0.1],
                "regressor__intercept_init": [0.2, 0.5],
            },
            {"regressor": [random_forest], "regressor__n_models": [10, 15]},
        ]

    def test_detect_and_increment(
        self, regression_pipeline, regression_models, regression_dataset
    ):
        X, y = regression_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=regression_pipeline,
            pipeline_params_grid=self.get_params_grid(regression_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "r2",
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

            auto_pipeline.train(X_train, y_train)

    def test_detect_and_retrain(
        self, regression_pipeline, regression_models, regression_dataset
    ):
        X, y = regression_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=regression_pipeline,
            pipeline_params_grid=self.get_params_grid(regression_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "r2",
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

            auto_pipeline.train(X_train, y_train)

    def test_detect_and_warm_start(
        self, regression_pipeline, regression_models, regression_dataset
    ):
        X, y = regression_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=regression_pipeline,
            pipeline_params_grid=self.get_params_grid(regression_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "r2",
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

            auto_pipeline.train(X_train, y_train)

    def test_detect_and_restart(
        self, regression_pipeline, regression_models, regression_dataset
    ):
        X, y = regression_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=regression_pipeline,
            pipeline_params_grid=self.get_params_grid(regression_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "r2",
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

            auto_pipeline.train(X_train, y_train)

    def test_detect_and_periodic_restart(
        self, regression_pipeline, regression_models, regression_dataset
    ):
        X, y = regression_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=regression_pipeline,
            pipeline_params_grid=self.get_params_grid(regression_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "r2",
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

            auto_pipeline.train(X_train, y_train)

    def test_prequential_training(
        self, regression_pipeline, regression_models, regression_dataset
    ):
        X, y = regression_dataset
        auto_pipeline = SAILAutoPipeline(
            pipeline=regression_pipeline,
            pipeline_params_grid=self.get_params_grid(regression_models),
            search_method=SAILTuneGridSearchCV,
            search_method_params={
                "verbose": 0,
                "num_cpus_per_trial": 1,
                "max_iters": 1,
                "early_stopping": False,
                "mode": "max",
                "scoring": "r2",
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

            auto_pipeline.train(X_train, y_train)
