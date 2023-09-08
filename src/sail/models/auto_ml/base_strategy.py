from enum import Enum, auto

import numpy as np
import pandas as pd
import ray

from sail.common.decorators import log_epoch
from sail.common.helper import VerboseManager
from sail.drift_detection.drift_detector import SAILDriftDetector
from sail.utils.logging import configure_logger

# from sail.visualisation.tensorboard import TensorboardWriter

LOGGER = configure_logger(logger_name="PipelineStrategy")
# writer = TensorboardWriter("/Logs")


class PipelineActionType(Enum):
    DATA_COLLECTION = auto()
    FIND_BEST_PIPELINE = auto()
    WARM_START_FIND_BEST_PIPELINE = auto()
    SCORE_AND_DETECT_DRIFT = auto()
    FIT_MODEL = auto()
    PARTIAL_FIT_MODEL = auto()
    PARTIAL_FIT_PIPELINE = auto()


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
        drift_detector=None,
        verbosity=VerboseManager(),
        incremental_training=False,
    ) -> None:
        self.search_method = search_method
        self.search_data_size = search_data_size
        self.drift_detector = drift_detector
        if drift_detector is None:
            self.drift_detector = SAILDriftDetector()
        self.verbosity = verbosity
        self.incremental_training = incremental_training

    def set_current_action(self, current_action: PipelineAction):
        self.pipeline_actions.current_action_node = current_action

    def get_current_action(self):
        return self.pipeline_actions.current_action_node

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
            if self.incremental_training:
                score = self._best_pipeline._progressive_score(X, y, detached=True)
            else:
                score = self._best_pipeline.score(X, y)

            y_pred = self._best_pipeline.predict(X)

            # write progress to tensorboard
            # writer.write_predictions(y_pred, y)
            # writer.write_score(score, self.verbosity.current_epoch_n)

            if (
                not self._detect_drift(score=score, y_pred=y_pred, y_true=y)
            ) and self.incremental_training:
                self._partial_fit_pipeline(X, y, **fit_params)
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
            fit_result = self.search_method.fit(
                X=self._input_X,
                y=self._input_y,
                warm_start=warm_start,
                tune_params=tune_params,
                **fit_params,
            )
        except Exception as e:
            LOGGER.error(e)
            ray.shutdown()
            raise Exception(
                "Pipeline tuning failed. Disconnecting Ray cluster. Please check logs."
            )
        finally:
            ray.shutdown()

        LOGGER.info("Pipeline tuning completed. Disconnecting Ray cluster...")

        # set best estimator and fit results
        self._best_pipeline = fit_result.best_estimator_
        self._best_pipeline._verbosity = self.verbosity
        self._fit_result = fit_result
        LOGGER.info(f"Found best params: {fit_result.best_params}")

        # housekeeping
        del self.__dict__["_input_X"]
        del self.__dict__["_input_y"]

        self.pipeline_actions.next()

    def _partial_fit_pipeline(self, X, y, **fit_params):
        self._best_pipeline._partial_fit(X, y, **fit_params)

    def _partial_fit_model(self, X, y, **fit_params):
        self._best_pipeline.fit_final_estimator(X, y, warm_start=True, **fit_params)
        self.pipeline_actions.next()

    def _fit_model(self, X, y, **fit_params):
        self._best_pipeline.fit_final_estimator(X, y, warm_start=False, **fit_params)
        self.pipeline_actions.next()

    def _detect_drift(self, **kwargs):
        self.drift_detector.set_verbose(self.verbosity.get())
        return self.detect_drift(**kwargs)
