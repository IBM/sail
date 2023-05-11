import importlib
from enum import Enum, auto
from typing import Union

import numpy as np
import pandas as pd
from ray.tune.sklearn import TuneGridSearchCV, TuneSearchCV
from river.base import DriftDetector
from river.drift import PageHinkley
from sklearn.utils import check_array

from sail.models.base import SAILModel
from sail.pipeline import SAILPipeline
from sail.utils.logging import configure_logger

LOGGER = configure_logger()


class PipelineState(Enum):
    NEW = auto()
    FITTING = auto()
    BEST = auto()
    STALE = auto()


class PipeLineSelection(SAILModel):
    def __init__(
        self,
        pipeline: SAILPipeline,
        pipeline_params_grid: dict,
        search_method: Union[str, TuneSearchCV, TuneGridSearchCV] = None,
        search_method_params: dict = None,
        search_data_size: int = 1000,
        mode: Union["min", "max"] = "max",
        scoring: str = "Accuracy",
        drift_detector: Union[str, DriftDetector] = "auto",
    ) -> None:
        self.pipeline = pipeline
        self.pipeline_params_grid = pipeline_params_grid
        self.search_data_size = search_data_size
        self.search_method = self._check_search_method(
            search_method, search_method_params, mode, scoring
        )
        self.mode = mode
        self.scoring = scoring
        self.cumulative_scorer = self._check_scoring(scoring)
        self.drift_detector = self._check_drift_detector(drift_detector)
        self._pipeline_state = PipelineState.NEW

    @property
    def best_pipeline(self) -> SAILPipeline:
        self.check_is_fitted("best_pipeline()")
        return self._best_pipeline

    @property
    def cumulative_score(self) -> float:
        self.check_is_fitted("cumulative_score()")
        return self.cumulative_scorer.get()

    @property
    def cv_results(self):
        self.check_is_fitted("cv_results()")
        if hasattr("fit_result", self):
            return self.fit_result.cv_results_

    def check_is_fitted(self, func) -> None:
        fitted = True
        if self._pipeline_state == PipelineState.FITTING:
            LOGGER.info(
                f"Ignoring...{func}, since the best pipeline is not available yet. Keep calling 'partial_fit' with new data to get the best pipeline."
            )
            fitted = False
        elif not hasattr(self, "_best_pipeline"):
            raise Exception(
                f"The current instance is not fitted yet. Call 'partial_fit' with appropriate arguments before using {func}."
            )
        elif self._pipeline_state == PipelineState.STALE:
            LOGGER.warning(
                f"The current best pipeline is STALE. Pipeline becomes stale when data drift occurs. You can call 'partial_fit' with fresh data to get the best pipeline. Please note that input data of length {self.search_data_size} is required to begin new search."
            )

        return fitted

    def _validate_is_2darray(self, X, y=None):
        X = check_array(X, ensure_2d=False)
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))
            if y is not None:
                y = check_array(
                    np.array(y, ndmin=1),
                    ensure_2d=False,
                )

        return X, y

    def _check_scoring(self, scoring):
        module = importlib.import_module("river.metrics")
        try:
            _scoring_class = getattr(module, scoring)
        except AttributeError:
            raise Exception(f"Method '{scoring}' is not available in River.metrics.")
        return _scoring_class()

    def _check_drift_detector(self, drift_detector) -> DriftDetector:
        if isinstance(drift_detector, DriftDetector):
            return drift_detector

        if drift_detector == "auto":
            _drift_detector_class = PageHinkley
        elif isinstance(drift_detector, str):
            module = importlib.import_module("river.drift")
            try:
                _drift_detector_class = getattr(module, drift_detector)
            except AttributeError:
                raise Exception(
                    f"Drift Detector '{drift_detector}' is not available in River."
                )

        return _drift_detector_class()

    def _check_search_method(self, search_method, search_method_params, mode, scoring):
        if search_method is None:
            _search_class = TuneGridSearchCV
        elif isinstance(search_method, str):
            module = importlib.import_module("ray.tune.sklearn")
            try:
                _search_class = getattr(module, search_method)
            except AttributeError:
                raise Exception(
                    f"Method '{search_method}' is not available in Ray Tune."
                )
        elif search_method in [TuneSearchCV, TuneGridSearchCV]:
            _search_class = search_method

        if search_method_params is None:
            search_method_params = {
                "max_iters": 1,
                "early_stopping": False,
                "mode": mode,
                "scoring": "accuracy2",
                "pipeline_auto_early_stop": False,
            }

        return _search_class(
            self.pipeline, self.pipeline_params_grid, **search_method_params
        )

    def partial_fit(self, X, y=None, **fit_params):
        """_summary_

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data.

        y : ndarray of shape (n_samples,)
            Subset of the target values.
        """

        X, y = self._validate_is_2darray(X, y)
        if not hasattr(self, "_best_pipeline") or self._pipeline_state == "STALE":
            if not hasattr(self, "_search_input_X"):
                self._search_input_X = X
                self._search_input_y = y
            else:
                self._search_input_X = np.vstack((self._search_input_X, X))
                self._search_input_y = np.hstack((self._search_input_y, y))

            LOGGER.info(
                f"Collecting data for pipeline tuning. Current Batch Size: {self._search_input_X.shape[0]}. Required: {self.search_data_size}"
            )
            self._pipeline_state = PipelineState.FITTING

            if self._search_input_X.shape[0] >= self.search_data_size:
                LOGGER.info(
                    f"Data collection completed for pipeline tuning. Final Batch Size: {self._search_input_X.shape[0]}."
                )
                self._find_best_pipeline(
                    self._search_input_X,
                    self._search_input_y,
                )
                self._pipeline_state = PipelineState.BEST
                del self.__dict__["_search_input_X"]
                del self.__dict__["_search_input_y"]
        else:
            self._detect_drift(self._cumulative_scoring(X, y))
            self._best_pipeline.partial_fit(X, y, **fit_params)

    def predict(self, X, **predict_params):
        if self.check_is_fitted("predict()"):
            X, _ = self._validate_is_2darray(X)
            return self.best_pipeline.predict(X, **predict_params)

    def _cumulative_scoring(self, X, y=None, sample_weight=1.0):
        X, y = self._validate_is_2darray(X, y)
        y_preds = self.predict(X)

        # Cumulative scoring
        for v1, v2 in zip(y, y_preds):
            self.cumulative_scorer.update(v1, v2, sample_weight)

        score = self.cumulative_scorer.get()
        LOGGER.info(f"Cumulative Pipeline Score: {score}")

        return score

    def score(self, X, y=None, sample_weight=1.0):
        if self.check_is_fitted("score()"):
            X, y = self._validate_is_2darray(X, y)
            y_preds = self.predict(X)

            scoring_metric = self._check_scoring(self.scoring)
            for v1, v2 in zip(y, y_preds):
                scoring_metric.update(v1, v2, sample_weight)

            score = scoring_metric.get()
            LOGGER.info(f"Pipeline Score: {score}")

            return score

    def _detect_drift(self, value):
        if self._pipeline_state == PipelineState.BEST:
            pass
        self.drift_detector.update(value)
        if self.drift_detector.drift_detected:
            LOGGER.info(
                "Drift Detected in the data. Pipeline will be re-tuned on the next partial_fit()"
            )
            self._pipeline_state = PipelineState.STALE

    def _find_best_pipeline(
        self, X, y=None, groups=None, tune_params=None, **fit_params
    ):
        LOGGER.info(f"Pipeline tuning using {self.search_method.__class__}")
        LOGGER.debug(f"Tuning params grid: {self.pipeline_params_grid}")

        fit_result = self.search_method.fit(X, y)
        self.fit_result = fit_result
        self._best_pipeline = fit_result.best_estimator_
