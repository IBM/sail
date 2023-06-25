from abc import ABC
from enum import Enum, auto
from typing import Union
import time
import numpy as np
from ray import tune
from ray.air import Checkpoint, session
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.tune.sklearn import TuneGridSearchCV, TuneSearchCV

from sail.utils.logging import configure_logger

LOGGER = configure_logger()


class PipelineActionType(Enum):
    DATA_COLLECTION = auto()
    FIND_BEST_PIPELINE = auto()
    SCORE_AND_DETECT_DRIFT = auto()
    FIT_MODEL = auto()
    PARTIAL_FIT_MODEL = auto()
    INCREMENTAL_TRAIN = auto()
    PARTIAL_FIT_BEST_PIPELINE = auto()


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
    ]

    def __init__(
        self,
        search_method,
        search_data_size,
        cumulative_scorer,
        drift_detector,
        incremental_training,
    ) -> None:
        self.search_method = search_method
        self.search_data_size = search_data_size
        self.cumulative_scorer = cumulative_scorer
        self.drift_detector = drift_detector
        self.incremental_training = incremental_training

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
            == PipelineActionType.PARTIAL_FIT_BEST_PIPELINE
        ):
            self._find_best_pipeline(warm_start=True, **fit_params)
        elif (
            self.pipeline_actions.current_action
            == PipelineActionType.SCORE_AND_DETECT_DRIFT
        ):
            self._cumulative_scoring(X, y)
            if self.incremental_training:
                self._incremental_train(X, y, **fit_params)
            self._detect_drift()
        elif (
            self.pipeline_actions.current_action == PipelineActionType.INCREMENTAL_TRAIN
        ):
            self._incremental_train(X, y, **fit_params)
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
        LOGGER.info(f"Pipeline tuning using {self.search_method.__class__}")
        tune_params.update(
            {
                "name": "SAILAutoML_Experiment"
                + "_"
                + time.strftime("%d-%m-%Y_%H:%M:%S"),
                "trial_dirname_creator": lambda trial: f"Trail_{trial.trial_id}",
            }
        )
        fit_result = self.search_method.fit(
            X=self._input_X,
            y=self._input_y,
            warm_start=warm_start,
            tune_params=tune_params,
            **fit_params,
        )
        self._fit_result = fit_result
        self._best_pipeline = fit_result.best_estimator_
        del self.__dict__["_input_X"]
        del self.__dict__["_input_y"]
        self.pipeline_actions.next()

    def _incremental_train(self, X, y, **fit_params):
        LOGGER.info("Partially fitting best pipeline.")
        self._best_pipeline.partial_fit(X, y, **fit_params)

    def _cumulative_scoring(self, X, y, sample_weight=1.0):
        y_preds = self._best_pipeline.predict(X)

        # Cumulative scoring
        for v1, v2 in zip(y, y_preds):
            self.cumulative_scorer.update(v1, v2, sample_weight)

        score = self.cumulative_scorer.get()
        LOGGER.info(f"Cumulative Pipeline Score: {score}")

    def _partial_fit_model(self, X, y, **fit_params):
        self._best_pipeline.fit_final_estimator(X, y, warm_start=True, **fit_params)
        self.pipeline_actions.next()

    def _fit_model(self, X, y, **fit_params):
        self._best_pipeline.fit_final_estimator(X, y, warm_start=False, **fit_params)
        self.pipeline_actions.next()

    def _detect_drift(self):
        value = self.cumulative_scorer.get()
        self.drift_detector.update(value)
        if self.drift_detector.drift_detected:
            LOGGER.info("Drift Detected in the data.")
            self.pipeline_actions.next()
