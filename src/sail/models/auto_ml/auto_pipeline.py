import importlib
import inspect
import os
import shutil
from typing import Type, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from sail.drift_detection.drift_detector import SAILDriftDetector
from sail.models.auto_ml.base_strategy import PipelineActionType, PipelineStrategy
from sail.models.auto_ml.pipeline_strategy import DetectAndIncrement
from sail.models.auto_ml.tune import SAILTuneGridSearchCV, SAILTuneSearchCV
from sail.models.base import SAILModel
from sail.pipeline import SAILPipeline
from sail.utils.logging import configure_logger
from sail.utils.serialization import load_obj, save_obj

LOGGER = configure_logger(logger_name="SAILAutoPipeline")


class SAILAutoPipeline(SAILModel, BaseEstimator):
    def __init__(
        self,
        pipeline: SAILPipeline,
        pipeline_params_grid: dict,
        search_method: Union[None, str] = None,
        search_method_params: dict = None,
        search_data_size: int = 1000,
        incremental_training: bool = False,
        drift_detector: Union[None, SAILDriftDetector] = None,
        pipeline_strategy: Union[None, str] = None,
        cluster_address: str = None,
    ) -> None:
        self.pipeline = pipeline
        self.pipeline_params_grid = pipeline_params_grid
        self.search_method_params = search_method_params
        self.search_data_size = search_data_size
        self.incremental_training = incremental_training
        self.drift_detector = drift_detector
        self.cluster_address = cluster_address

        self.search_method = self._resolve_search_method(
            search_method, search_method_params, cluster_address
        )
        self.pipeline_strategy = self._resolve_pipeline_strategy(
            pipeline_strategy, incremental_training
        )

    @property
    def best_pipeline(self) -> SAILPipeline:
        if hasattr(self.pipeline_strategy, "_best_pipeline"):
            return self.pipeline_strategy._best_pipeline
        return None

    @property
    def progressive_score(self) -> float:
        if self.check_is_fitted("progressive_score"):
            return self.best_pipeline.get_progressive_score

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
            additional_msg = (
                " Please note that input data of length {self.search_data_size} is required to begin new parameter search."
                if self.pipeline_strategy
                in ["DetectAndWarmStart", "DetectAndRestart", "PrequentialTraining"]
                else ""
            )
            LOGGER.warning(
                f"The current best pipeline is STALE. Pipeline becomes stale when data drift occurs. You can call 'train' with fresh data to get the best pipeline."
                + additional_msg
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

    def _resolve_search_method(
        self, search_method, search_method_params, cluster_address
    ):
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
            estimator=self.pipeline,
            param_grid=self.pipeline_params_grid,
            cluster_address=cluster_address,
            **search_method_params,
        )

    def _resolve_pipeline_strategy(self, pipeline_strategy, incremental_training):
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
            search_method=self.search_method,
            search_data_size=self.search_data_size,
            drift_detector=self.drift_detector,
            incremental_training=incremental_training,
        )

    def train(self, X, y=None, **fit_params):
        X, y = self._validate_is_2darray(X, y)
        self.pipeline_strategy.next(X, y, **fit_params)

    def predict(self, X, **predict_params):
        if self.check_is_fitted("predict()"):
            X, _ = self._validate_is_2darray(X)
            return self.best_pipeline.predict(X, **predict_params)

    def score(self, X, y=None, sample_weight=1.0) -> float:
        if self.check_is_fitted("score()"):
            return self.best_pipeline.score(X, y, sample_weight)

    def save_model(self, model_folder, overwrite=True) -> str:
        """
        Parameters:
        -----------
        model_folder: str
            location to save the model
        overwrite: bool (False)
            if True and model_folder already exists, it will be delted and recreated

        Returns
        -------
        saved_location: str
        """
        save_location = os.path.join(model_folder, "sail_auto_pipeline")
        if not overwrite and os.path.exists(save_location):
            raise Exception(
                f"{save_location} already exists, specify overwrite=True to replace contents"
            )
        else:
            # delete old folder to avoid unwanted file overlaps
            if os.path.exists(save_location) and os.path.isdir(save_location):
                shutil.rmtree(save_location)

            LOGGER.info(f"making directory tree {save_location}")
            os.makedirs(save_location, exist_ok=True)
            if not os.path.exists(save_location):
                raise Exception(f"target directory {save_location} can not be created!")

        # -------------------------------------------
        # Get all params
        # -------------------------------------------
        params = self.get_params(deep=False)

        # -------------------------------------------
        # save search_method params
        # -------------------------------------------
        obj = []
        search_method = params.pop("search_method")
        if hasattr(search_method, "_best_configurations"):
            obj = search_method._best_configurations
        save_obj(
            obj,
            location=os.path.join(save_location, "search_method"),
            file_name="best_configurations",
        )

        # -------------------------------------------
        # save pipeline_strategy state
        # -------------------------------------------
        pipeline_strategy = params.pop("pipeline_strategy")
        save_obj(
            {"current_action": pipeline_strategy.get_current_action()},
            location=os.path.join(save_location, "pipeline_strategy"),
            file_name="state",
        )

        # -------------------------------------------
        # save data already collected for auto ml tuning
        # -------------------------------------------
        if hasattr(pipeline_strategy, "_input_X"):
            np.savez(
                os.path.join(save_location, "pipeline_strategy", "data"),
                input_X=pipeline_strategy._input_X,
                input_y=pipeline_strategy._input_y,
            )

        # -------------------------------------------
        # save fit results and best pipeline
        # -------------------------------------------
        if hasattr(pipeline_strategy, "_fit_result"):
            save_obj(
                pipeline_strategy._fit_result,
                location=os.path.join(save_location, "pipeline_strategy"),
                file_name="fit_result",
            )
            pipeline_strategy._best_pipeline.save(
                os.path.join(save_location, "pipeline_strategy"), name="best_pipeline"
            )

        # -------------------------------------------
        # save rest of the params
        # -------------------------------------------
        save_obj(
            obj=params,
            location=save_location,
            file_name="params",
        )

        return save_location

    @classmethod
    def load_model(cls, model_folder):
        load_location = os.path.join(model_folder, "sail_auto_pipeline")

        # -------------------------------------------
        # Load params
        # -------------------------------------------
        params = load_obj(location=load_location, file_name="params")

        # -------------------------------------------
        # create SAILAutoPipeline
        # -------------------------------------------
        sail_auto_pipeline = SAILAutoPipeline(**params)

        # -------------------------------------------
        # Load best_configurations if present
        # -------------------------------------------
        best_configurations = load_obj(
            location=os.path.join(load_location, "search_method"),
            file_name="best_configurations",
        )
        if len(best_configurations) > 0:
            sail_auto_pipeline.search_method._best_configurations = best_configurations

        # -------------------------------------------
        # Load existing pipeline_strategy state
        # -------------------------------------------
        state = load_obj(
            location=os.path.join(load_location, "pipeline_strategy"),
            file_name="state",
        )
        sail_auto_pipeline.pipeline_strategy.set_current_action(state["current_action"])

        # -------------------------------------------
        # Load data already collected data for auto ml tuning
        # -------------------------------------------
        if os.path.exists(os.path.join(load_location, "pipeline_strategy", "data.npz")):
            data = np.load(os.path.join(load_location, "pipeline_strategy", "data.npz"))
            sail_auto_pipeline.pipeline_strategy._input_X = data["input_X"]
            sail_auto_pipeline.pipeline_strategy._input_y = data["input_y"]
            data.close()

        # -------------------------------------------
        # Load fit results and best pipeline
        # -------------------------------------------
        try:
            fit_result = load_obj(
                location=os.path.join(load_location, "pipeline_strategy"),
                file_name="fit_result",
            )
            sail_auto_pipeline.pipeline_strategy._fit_result = fit_result
            best_pipeline = SAILPipeline.load(
                os.path.join(load_location, "pipeline_strategy"), name="best_pipeline"
            )
            sail_auto_pipeline.pipeline_strategy._best_pipeline = best_pipeline
        except Exception as e:
            print(e)

        return sail_auto_pipeline
