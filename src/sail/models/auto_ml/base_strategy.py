from enum import Enum, auto

import numpy as np
import pandas as pd
import ray
from sklearn.base import clone

from sail.common.decorators import log_epoch
from sail.telemetry import trace_with_action
from sail.utils.logging import configure_logger
from sail.visualisation.tensorboard import TensorboardWriter

LOGGER = configure_logger(logger_name="PipelineStrategy")


class PipelineActionType(Enum):
    DATA_COLLECTION = auto()
    FIND_BEST_PIPELINE = auto()
    WARM_START_FIND_BEST_PIPELINE = auto()
    SCORE_AND_DETECT_DRIFT = auto()
    FIT_MODEL = auto()
    PARTIAL_FIT_MODEL = auto()
    PARTIAL_FIT_PIPELINE = auto()
    DRIFT_DETECTION = auto()


class PipelineAction:
    def __init__(self, action):
        self.action = action
        self.next = None
        self.previous = None


class PipelineActions:
    def __init__(self):
        self.head_action_node = None
        self.current_action_node = None

    @property
    def current_action(self):
        return self.current_action_node.action

    def get_action_node(self, action):
        if action:
            action_node = self.head_action_node
            while action_node is not None:
                if action_node.action == action:
                    return action_node
                action_node = action_node.next
        return None

    def add_action(self, action, next=None):
        new_action_node = PipelineAction(action)
        if self.head_action_node is None:
            self.head_action_node = new_action_node
            self.current_action_node = self.head_action_node
            return

        last_action_node = self.head_action_node
        while last_action_node.next:
            last_action_node = last_action_node.next

        last_action_node.next = new_action_node
        new_action_node.previous = last_action_node
        new_action_node.next = self.get_action_node(next)

    def get_actions(self):
        actions = []
        action_node = self.head_action_node
        while action_node is not None:
            if action_node.action.name in actions:
                break
            actions.append(action_node.action.name)
            action_node = action_node.next
        return actions

    def next(self):
        self.current_action_node = self.current_action_node.next


class PipelineStrategy:
    defined_stategies = [
        "DetectAndIncrement",
        "DetectAndRetrain",
        "DetectAndWarmStart",
        "DetectAndRestart",
        "PeriodicRestart",
        "PrequentialTraining",
    ]

    def __init__(
        self,
        search_method,
        search_data_size,
        drift_detector,
        verbosity,
        incremental_training=False,
        tensorboard_log_dir=None,
        tracer=None,
    ) -> None:
        self.search_method = search_method
        self.search_data_size = search_data_size
        self.drift_detector = drift_detector
        self.verbosity = verbosity
        self.incremental_training = incremental_training
        self.tensorboard_log_dir = tensorboard_log_dir
        self.tracer = tracer

    def set_current_action(self, current_action: PipelineAction):
        self.pipeline_actions.current_action_node = current_action

    def get_current_action(self):
        return self.pipeline_actions.current_action_node

    def get_stats_writer(self):
        # fmt: off
        assert self.tensorboard_log_dir is not None, "Parameter 'tensorboard_log_dir' is None. Unable to create stats writer."
        
        if not hasattr(self, "writer"):
            self.writer = TensorboardWriter(self.tensorboard_log_dir)
        return self.writer

    @log_epoch
    def next(self, X, y=None, tune_params={}, **fit_params):
        if self.pipeline_actions.current_action == PipelineActionType.DATA_COLLECTION:
            self._collect_data_for_parameter_tuning(X, y)

        if (
            self.pipeline_actions.current_action
            == PipelineActionType.FIND_BEST_PIPELINE
        ):
            self._find_best_pipeline(**fit_params)

        elif (
            self.pipeline_actions.current_action
            == PipelineActionType.WARM_START_FIND_BEST_PIPELINE
        ):
            self._find_best_pipeline(warm_start=True, **fit_params)

        elif (
            self.pipeline_actions.current_action
            == PipelineActionType.SCORE_AND_DETECT_DRIFT
        ):
            self._score_and_detect_drift(X, y, **fit_params)

        elif (
            self.pipeline_actions.current_action
            == PipelineActionType.PARTIAL_FIT_PIPELINE
        ):
            self._partial_fit_pipeline(X, y, **fit_params)

        elif (
            self.pipeline_actions.current_action == PipelineActionType.PARTIAL_FIT_MODEL
        ):
            self._partial_fit_model(X, y, **fit_params)

        elif self.pipeline_actions.current_action == PipelineActionType.FIT_MODEL:
            self._fit_model(X, y, **fit_params)

    @trace_with_action(PipelineActionType.DATA_COLLECTION.name)
    def _collect_data_for_parameter_tuning(self, X, y):
        if not hasattr(self, "_input_X"):
            self._input_X = X
            self._input_y = y
        else:
            if isinstance(self._input_X, pd.DataFrame):
                self._input_X = pd.concat([self._input_X, X])
            else:
                self._input_X = np.vstack((self._input_X, X))
            self._input_y = np.hstack((self._input_y, y))

        if self._input_X.shape[0] < self.search_data_size:
            if self.verbosity.get() == 1:
                LOGGER.info(
                    f"Collecting data for pipeline tuning. Current Batch Size: {self._input_X.shape[0]}. Required: {self.search_data_size}"
                )
        else:
            LOGGER.info(
                f"Data collection completed for pipeline tuning. Final Batch Size: {self._input_X.shape[0]}."
            )
            self.pipeline_actions.next()

    def _find_best_pipeline(self, tune_params={}, warm_start=False, **fit_params):
        class_name = (
            self.search_method.__class__.__module__
            + "."
            + self.search_method.__class__.__qualname__
        )
        LOGGER.info(f"Starting Pipeline tuning with {class_name}")
        ray.init(
            address=self.search_method.cluster_address,
            namespace=self.search_method.namespace,
            runtime_env=self.search_method.runtime_env,
        )
        LOGGER.info(
            f"Cluster resources: Nodes: {len(ray.nodes())}, Cluster CPU: {ray.cluster_resources()['CPU']}, Cluster Memory: {str(format(ray.cluster_resources()['memory'] / (1024 * 1024 * 1024), '.2f')) + ' GB'}"
        )
        try:
            best_estimator, best_params = self._tune_pipeline(
                tune_params, warm_start, **fit_params
            )
        except Exception as e:
            raise Exception(
                f"Pipeline tuning failed. Disconnecting Ray cluster. {str(e)}"
            )
        finally:
            ray.shutdown()

        LOGGER.info("Pipeline tuning completed. Disconnecting Ray cluster...")

        # set best estimator and fit results
        # self._fit_result = fit_result
        self._best_pipeline = best_estimator

        # replace verbosity instance of the best pipeline from the parent.
        self._best_pipeline.verbosity = self.verbosity

        LOGGER.info(f"Found best params: {best_params}")

        # housekeeping
        del self.__dict__["_input_X"]
        del self.__dict__["_input_y"]

        self.pipeline_actions.next()

    @trace_with_action(PipelineActionType.FIND_BEST_PIPELINE.name)
    def _tune_pipeline(self, tune_params, warm_start, **fit_params):
        fit_result = self.search_method.fit(
            X=self._input_X,
            y=self._input_y,
            warm_start=warm_start,
            tune_params=tune_params,
            **fit_params,
        )
        if fit_result is None:
            raise Exception(
                f"The result of the pipeline tuning is None. Please check input features and search parameters of the search algorithm."
            )
        best_params = fit_result.best_params

        # prepare and fit best estimator
        best_estimator = clone(
            clone(self.search_method.estimator).set_params(**best_params)
        )
        best_estimator.verbosity.disabled()
        if self._input_y is not None:
            best_estimator.fit(self._input_X, self._input_y, **fit_params)
        else:
            best_estimator.fit(self._input_X, **fit_params)

        return best_estimator, best_params

    @trace_with_action(
        PipelineActionType.SCORE_AND_DETECT_DRIFT.name, current_span=True
    )
    def _score_and_detect_drift(self, X, y, **fit_params):
        if self.incremental_training:
            score = self._best_pipeline._progressive_score(X, y, detached=True)
        else:
            score = self._best_pipeline.score(X, y)

        y_pred = self._best_pipeline.predict(X)
        is_drift_detected = self._detect_drift(score=score, y_pred=y_pred, y_true=y)

        # write progress to tensorboard
        if self.tensorboard_log_dir:
            if self._best_pipeline._final_estimator._estimator_type == "classifier":
                self.get_stats_writer().write_classification_report(
                    y_pred, y, epoch_n=self.verbosity.current_epoch_n
                )
            else:
                start_index = self.verbosity.samples_seen_n - len(y_pred)
                self.get_stats_writer().write_predictions(
                    y_pred, y, start_index=start_index
                )
            self.get_stats_writer().write_score(
                score=score,
                epoch_n=self.verbosity.current_epoch_n,
                drift_point=is_drift_detected,
            )

        if (not is_drift_detected) and self.incremental_training:
            self._partial_fit_pipeline(X, y, **fit_params)

    @trace_with_action(PipelineActionType.PARTIAL_FIT_PIPELINE.name)
    def _partial_fit_pipeline(self, X, y, **fit_params):
        self._best_pipeline._partial_fit(X, y, **fit_params)

    @trace_with_action(PipelineActionType.PARTIAL_FIT_MODEL.name)
    def _partial_fit_model(self, X, y, **fit_params):
        self._best_pipeline.fit_final_estimator(X, y, warm_start=True, **fit_params)
        self.pipeline_actions.next()

    @trace_with_action(PipelineActionType.FIT_MODEL.name)
    def _fit_model(self, X, y, **fit_params):
        self._best_pipeline.fit_final_estimator(X, y, warm_start=False, **fit_params)
        self.pipeline_actions.next()

    @trace_with_action(PipelineActionType.DRIFT_DETECTION.name)
    def _detect_drift(self, **kwargs):
        self.drift_detector.set_verbose(self.verbosity.get())
        return self.detect_drift(**kwargs)
