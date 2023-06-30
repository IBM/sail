from tabnanny import verbose
import time
from abc import ABC
from enum import Enum, auto
from time import sleep
from typing import Union

import numpy as np
import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.tune.sklearn import TuneGridSearchCV, TuneSearchCV
from threading import Thread
from sail.utils.logging import configure_logger
from sail.utils.progress_bar import SAILProgressBar
from threading import Event

LOGGER = configure_logger()


def start_progress(params, event):
    progress = SAILProgressBar(
        steps=100,
        desc=f"SAIL Pipeline Tuning in progress...",
        params=params,
        format="tuning",
        verbose=1,
    )

    while True:
        progress.update()
        sleep(0.1)
        # check for stop
        if event.is_set():
            progress.finish()
            progress.close()
            break


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
    ]

    def __init__(
        self,
        search_method,
        search_data_size,
        drift_detector,
        incremental_training,
    ) -> None:
        self.search_method = search_method
        self.search_data_size = search_data_size
        self.drift_detector = drift_detector
        self.incremental_training = incremental_training

    def action_separator(self):
        print(
            ">>>--------------------------------------------------------------------------------------------"
        )

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
            self.action_separator()
            y_preds = self._best_pipeline.predict(X)
            if self.incremental_training:
                self._best_pipeline._scorer._eval_progressive_score(
                    y_preds, y, detached=True
                )
            else:
                self._best_pipeline.score(X, y)

            if not self._detect_drift(y_preds, y) and self.incremental_training:
                self._partial_fit_pipeline(X, y, **fit_params)
        elif (
            self.pipeline_actions.current_action
            == PipelineActionType.PARTIAL_FIT_PIPELINE
        ):
            self.action_separator()
            self._partial_fit_pipeline(X, y, **fit_params)
        elif (
            self.pipeline_actions.current_action == PipelineActionType.PARTIAL_FIT_MODEL
        ):
            self.action_separator()
            if self.incremental_training:
                y_preds = self._best_pipeline.predict(X)
                self._best_pipeline._scorer._eval_progressive_score(
                    y_preds, y, verbose=0
                )
            self._partial_fit_model(X, y, **fit_params)
        elif self.pipeline_actions.current_action == PipelineActionType.FIT_MODEL:
            self.action_separator()
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
        class_name = (
            self.search_method.__class__.__module__
            + "."
            + self.search_method.__class__.__qualname__
        )
        LOGGER.info(f"Starting Pipeline tuning with {class_name}")
        tune_params.update(
            {
                "trial_dirname_creator": lambda trial: f"Trail_{trial.trial_id}",
            }
        )
        ray.init()
        resources = ray.cluster_resources()
        event = Event()
        params = {
            "CPU": resources["CPU"],
            "Memory": resources["memory"] / (1024 * 1024 * 1024),
            "Class": self.search_method.__class__.__qualname__,
        }
        thread = Thread(
            target=start_progress,
            args=(
                params,
                event,
            ),
        )
        thread.start()
        fit_result = self.search_method.fit(
            X=self._input_X,
            y=self._input_y,
            warm_start=warm_start,
            tune_params=tune_params,
            **fit_params,
        )
        event.set()
        thread.join()
        LOGGER.info("Pipeline tuning completed. Shutting down Ray cluster...")
        ray.shutdown()
        self._fit_result = fit_result
        LOGGER.info(f"Found best params: {fit_result.best_params}")
        self._best_pipeline = fit_result.best_estimator_
        self._best_pipeline.log_verbose = 1
        del self.__dict__["_input_X"]
        del self.__dict__["_input_y"]
        self.pipeline_actions.next()

    def _partial_fit_pipeline(self, X, y, **fit_params):
        LOGGER.debug("Partially fitting best pipeline.")
        self._best_pipeline.partial_fit(X, y, **fit_params)

    def _partial_fit_model(self, X, y, **fit_params):
        self._best_pipeline.fit_final_estimator(
            X, y, warm_start=True, verbose=1, **fit_params
        )
        self.pipeline_actions.next()

    def _fit_model(self, X, y, **fit_params):
        self._best_pipeline.fit_final_estimator(X, y, warm_start=False, **fit_params)
        self.pipeline_actions.next()

    def _detect_drift(self, y_preds, y_true):
        if self.drift_detector.detect_drift(y_preds, y_true):
            LOGGER.info("Drift Detected in the data.")
            self.pipeline_actions.next()
