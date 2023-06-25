import importlib
import inspect
import sys
from typing import List, Type, Union

import numpy as np
import river
from river import metrics
from river.base import DriftDetector
from river.drift import PageHinkley
from sklearn.utils import check_array

from sail.models.auto_ml.base_strategy import PipelineActionType, PipelineStrategy
from sail.models.auto_ml.pipeline_strategy import DetectAndIncrement
from sail.models.auto_ml.tune import SAILTuneGridSearchCV, SAILTuneSearchCV
from sail.models.base import SAILModel
from sail.pipeline import SAILPipeline
from sail.utils.logging import configure_logger

LOGGER = configure_logger()


class SAILAutoPipeline(SAILModel):
    def __init__(
        self,
        pipeline: SAILPipeline,
        pipeline_params_grid: dict,
        search_method: Union[None, str] = None,
        search_method_params: dict = None,
        search_data_size: int = 1000,
        incremental_training: bool = False,
        scoring: Union[None, str, metrics.base.Metric] = None,
        drift_detector: Union[str, DriftDetector] = "auto",
        pipeline_strategy: Union[None, str] = None,
    ) -> None:
        self.scoring = scoring
        self.pipeline = pipeline
        self.pipeline_params_grid = pipeline_params_grid
        self.search_data_size = search_data_size
        self.search_method = self._check_search_method(
            search_method, search_method_params, scoring
        )
        self.incremental_training = incremental_training
        self.cumulative_scorer = self._check_scoring(scoring)
        self.drift_detector = self._check_drift_detector(drift_detector)
        self.pipeline_strategy = self.resolve_pipeline_strategy(pipeline_strategy)

    @property
    def best_pipeline(self) -> SAILPipeline:
        if hasattr(self.pipeline_strategy, "_best_pipeline"):
            return self.pipeline_strategy._best_pipeline
        return None

    @property
    def cumulative_score(self) -> float:
        self.check_is_fitted("cumulative_score()")
        return self.cumulative_scorer.get()

    @property
    def cv_results(self):
        self.check_is_fitted("cv_results()")
        if hasattr("fit_result", self):
            return self.pipeline_strategy._fit_result.cv_results_

    def check_is_fitted(self, func) -> None:
        pipeline_action = self.pipeline_strategy.pipeline_actions.current_action_node
        if pipeline_action.action == PipelineActionType.DATA_COLLECTION:
            LOGGER.warning(
                f"Ignoring...{func}, since the best pipeline is not available yet. Keep calling 'train' with new data to get the best pipeline."
            )
            fitted = False
        elif (
            pipeline_action.previous.action == PipelineActionType.SCORE_AND_DETECT_DRIFT
            and hasattr(self.pipeline_strategy, "_best_pipeline")
        ):
            LOGGER.warning(
                f"The current best pipeline is STALE. Pipeline becomes stale when data drift occurs. You can call 'partial_fit' with fresh data to get the best pipeline. Please note that input data of length {self.search_data_size} is required to begin new parameter search."
            )
            fitted = True
        elif not hasattr(self.pipeline_strategy, "_best_pipeline"):
            raise Exception(
                f"The current instance is not fitted yet. Call 'train' with appropriate arguments before using {func}."
            )
        elif hasattr(self.pipeline_strategy, "_best_pipeline"):
            fitted = True

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
        if scoring is None:
            return metrics.Accuracy()
        try:
            if isinstance(scoring, str):
                module = importlib.import_module("river.metrics")
                _scoring_class = getattr(module, scoring)
                return _scoring_class()
            elif isinstance(scoring, metrics.base.Metric):
                return scoring
            elif inspect.isclass(scoring):
                _scoring_class = scoring
                valid_classes = [
                    class_name
                    for _, class_name in inspect.getmembers(
                        sys.modules["river.metrics"], inspect.isclass
                    )
                ]
                if _scoring_class in valid_classes:
                    return _scoring_class()
                else:
                    raise Exception
            else:
                raise Exception

        except:
            method_name = (
                scoring.__name__
                if inspect.isclass(scoring)
                else scoring
                if isinstance(scoring, str)
                else scoring.__class__.__name__
            )
            raise AttributeError(
                f"Method '{method_name}' is not available in river.metrics. Scoring must be a str or an instance of the {river.metrics.__all__}."
            )

    def _check_drift_detector(self, drift_detector) -> DriftDetector:
        if isinstance(drift_detector, DriftDetector):
            return drift_detector
        elif isinstance(drift_detector, str):
            if drift_detector == "auto":
                _drift_detector_class = PageHinkley
            elif isinstance(drift_detector, str):
                module = importlib.import_module("river.drift")
                try:
                    _drift_detector_class = getattr(module, drift_detector)
                except AttributeError:
                    raise Exception(
                        f"Drift Detector '{drift_detector}' is not available in River. Available drift detectors: {river.drift.__all__}"
                    )
        else:
            raise TypeError(
                "`drift_detector` must be an instance or str from "
                f"{river.drift.__all__} from river.drift module. Got {drift_detector.__module__}.{drift_detector.__qualname__}. Set `auto` to use the default."
            )
        return _drift_detector_class()

    def _check_search_method(self, search_method, search_method_params, scoring):
        if search_method is None:
            _search_class = SAILTuneGridSearchCV
        elif Type[search_method] in [
            Type[SAILTuneGridSearchCV],
            Type[SAILTuneSearchCV],
        ]:
            _search_class = search_method
        elif isinstance(search_method, str):
            module = importlib.import_module("sail.models.auto_ml.tune")
            try:
                _search_class = getattr(module, search_method)
            except AttributeError:
                raise Exception(
                    f"Method '{search_method}' is not available. search_method must be from [SAILTuneGridSearchCV, SAILTuneSearchCV] from the module sail.models.auto_ml.tune. Set `None` to use the default."
                )
        else:
            raise TypeError(
                "`search_method` must be None or an instance or str from "
                f"[SAILTuneGridSearchCV, SAILTuneSearchCV] from sail.models.auto_ml.tune. Got {search_method.__module__}.{search_method.__qualname__}. Set `None` to use the default."
            )

        if search_method_params is None:
            search_method_params = _search_class.default_search_params
        else:
            # update params from the default ones if missing any.
            search_method_params = {
                **_search_class.default_search_params,
                **search_method_params,
            }

        return _search_class(
            self.pipeline, self.pipeline_params_grid, **search_method_params
        )

    def resolve_pipeline_strategy(self, pipeline_strategy):
        pipeline_strategy_class = None
        if pipeline_strategy is None:
            pipeline_strategy_class = DetectAndIncrement
        elif isinstance(pipeline_strategy, str):
            if pipeline_strategy in PipelineStrategy.defined_stategies:
                module = importlib.import_module(
                    "sail.models.auto_ml.pipeline_strategy"
                )
                pipeline_strategy_class = getattr(module, pipeline_strategy)
            else:
                raise ValueError(
                    "{} is not a defined pipeline strategy. "
                    "Please select from the list of available strategies: {}".format(
                        pipeline_strategy, PipelineStrategy.defined_stategies
                    )
                )
        else:
            raise TypeError(
                "`pipeline_strategy` must be a None, str, "
                f"or an instance of PipelineStrategy. Got {pipeline_strategy.__module__}.{pipeline_strategy.__qualname__}. Set `None` to use the default."
            )

        return pipeline_strategy_class(
            self.search_method,
            self.search_data_size,
            self.cumulative_scorer,
            self.drift_detector,
            incremental_training=self.incremental_training,
        )

    def train(self, X, y=None, **fit_params):
        X, y = self._validate_is_2darray(X, y)
        # if self.incremental_training:
        #     if not [param for param in fit_params if param.endswith("__classes")]:
        #         raise Exception(
        #             "If incremental training is enabled, train() must contain the classes parameter i.e. a list of all eligible classes. You can use the stepname__parameter format, e.g. `SAILAutoPipeline.train(X, y, classifier__classes=[1, 0])` where classifier is the stepname of the estimator."
        #         )
        self.pipeline_strategy.next(X, y, **fit_params)

    def predict(self, X, **predict_params):
        if self.check_is_fitted("predict()"):
            X, _ = self._validate_is_2darray(X)
            return self.best_pipeline.predict(X, **predict_params)

    def score(self, X, y=None, sample_weight=1.0):
        if self.check_is_fitted("score()"):
            X, y = self._validate_is_2darray(X, y)
            y_preds = self.predict(X)

            scoring_metric = self._check_scoring(self.scoring)
            for v1, v2 in zip(y, y_preds):
                scoring_metric.update(v1, v2, sample_weight)

            score = scoring_metric.get()
            return score
