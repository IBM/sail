import os
import shutil
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time

from sail.utils.logging import configure_logger
from sail.utils.progress_bar import SAILProgressBar
from sail.utils.scorer import SAILModelScorer
from sail.utils.serialization import load_obj, save_obj

LOGGER = configure_logger(logger_name="SAILPipeline")


class SAILPipeline(Pipeline):
    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        scoring=None,
        verbosity: int = 1,
    ):
        """[summary]

        Args:
            steps (list, optional): List of steps for the pipeline, required. Each step is a tuple containing a string, representing the name of a step, and a python object for the step.
            scoring: pipeline score metric
            verbosity: verbosity level for training logs. 0 (No logs) is default.
        """
        super(SAILPipeline, self).__init__(steps, verbose=False)
        self.scoring = scoring
        self._scorer = self._validate_and_get_scorer(scoring, steps[-1][1])
        self.verbosity = verbosity

    def _validate_and_get_scorer(self, scoring, estimator):
        if scoring is None:
            estimator_type = (
                None if estimator == "passthrough" else estimator._estimator_type
            )
            assert (
                estimator_type is not None
            ), "SAILPipeline.scoring cannot be None when the estimator is set to passthrough in SAILPipeline.steps."

        return SAILModelScorer(
            scoring=scoring,
            estimator_type=None
            if estimator == "passthrough"
            else estimator._estimator_type,
            pipeline_mode=True,
        )

    @property
    def get_progressive_score(self):
        return self._scorer.get_progressive_score

    def score(self, X, y, sample_weight=1.0):
        y_preds = self.predict(X)
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        return self._scorer.score(y, y_preds, sample_weight, verbose=self.verbosity)

    def progressive_score(self, X, y, sample_weight=1.0, detached=False, verbose=1):
        y_pred = self.predict(X)
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        return self._scorer.progressive_score(
            y, y_pred, sample_weight, detached, verbose
        )

    def fit(self, X, y=None, **fit_params):
        self._fit(X, y, warm_start=False, **fit_params)

    def partial_fit(self, X, y=None, **fit_params):
        if self.__sklearn_is_fitted__():
            self.progressive_score(X, y, verbose=0)
        self._fit(X, y, warm_start=True, **fit_params)

    def _fit(self, X, y=None, warm_start=None, **fit_params):
        Xh = X.copy()
        yh = y.copy()
        fit_params_steps = self._check_fit_params(**fit_params)
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = utils.validation.check_memory(self.memory)

        def _fit_transform_one(transformer, X, y, **fit_params):
            if warm_start and hasattr(transformer, "partial_fit"):
                transformed_X = transformer.partial_fit(X, y, **fit_params).transform(X)
            elif hasattr(transformer, "fit_transform"):
                transformed_X = transformer.fit_transform(X, y, **fit_params)
            else:
                transformed_X = transformer.fit(X, y, **fit_params).transform(X)
            return transformed_X, transformer

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        with SAILProgressBar(
            steps=len(list(self._iter(with_final=True, filter_passthrough=True))),
            desc=f"SAIL Pipeline Partial fit" if warm_start else f"SAIL Pipeline fit",
            params={
                "Batch Size": X.shape[0],
            },
            format="pipeline_training",
            verbose=self.verbosity,
        ) as progress:
            for step_idx, name, transformer in self._iter(
                with_final=False, filter_passthrough=True
            ):
                progress.append_desc(f"[{name}]")
                if transformer is None or transformer == "passthrough":
                    with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                        continue

                if hasattr(memory, "location") and memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transformer
                X, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    **fit_params_steps[name],
                )
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)

                progress.update()

            with _print_elapsed_time(
                "Pipeline", self._log_message(len(self.steps) - 1)
            ):
                if self._final_estimator != "passthrough":
                    progress.append_desc(f"[{self.steps[-1][0]}]")
                    self.fit_final_estimator(
                        X, y, warm_start=warm_start, **fit_params_steps
                    )
                progress.update()

            # update score for fit() call.
            if not warm_start:
                progress.update_params("Score", self.score(Xh, yh))
            else:
                progress.update_params("P_Score", self.get_progressive_score)

        return self

    def fit_final_estimator(
        self, X, y, warm_start=False, verbose=0, **fit_params_steps
    ):
        with SAILProgressBar(
            steps=1,
            desc=f"SAIL Model Partial fit" if warm_start else f"SAIL Model fit",
            params={
                "Model": self.steps[-1][1].__class__.__name__,
                "Batch Size": X.shape[0],
            },
            format="model_training",
            verbose=verbose,
        ) as progress:
            if self.steps[-1][0] in fit_params_steps:
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            else:
                fit_params_last_step = {}

            if warm_start:
                if not hasattr(self._final_estimator, "partial_fit"):
                    raise AttributeError(
                        f"Final Estimator: '{self.steps[-1][0]}' does not implement partial_fit()."
                    )
                self._final_estimator.partial_fit(X, y, **fit_params_last_step)
                progress.update()
                progress.update_params("P_Score", self.get_progressive_score)
            else:
                if not hasattr(self._final_estimator, "fit"):
                    raise AttributeError(
                        f"Final Estimator '{self.steps[-1][0]}' does not implement fit()."
                    )
                if "classes" in fit_params_last_step:
                    fit_params_last_step.pop("classes")
                self._final_estimator.fit(X, y, **fit_params_last_step)
                progress.update()
                progress.update_params("Score", self.score(X, y))

    def save(self, model_folder, name="sail_pipeline", overwrite=True) -> str:
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
        save_location = os.path.join(model_folder, name)
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
        # explicity save all the steps and steps names
        # -------------------------------------------
        for i, (step_name, step) in enumerate(self.steps):
            save_obj(
                obj=step,
                location=os.path.join(save_location, "steps"),
                file_name=step_name,
            )
        save_obj(
            obj=[step_name for (step_name, _) in self.steps],
            location=os.path.join(save_location, "steps"),
            file_name="steps_meta",
            serialize_type="json",
        )

        # -------------------------------------------
        # save scorer progressive state if present
        # -------------------------------------------
        if hasattr(self._scorer, "_y_true"):
            np.savez(
                os.path.join(save_location, "scorer_state"),
                y_true=self._scorer._y_true,
                y_pred=self._scorer._y_pred,
            )

        # -------------------------------------------
        # save rest of the params
        # -------------------------------------------
        params = self.get_params(deep=False)
        params.pop("steps")
        save_obj(
            obj=params,
            location=save_location,
            file_name="params",
        )

        return save_location

    @classmethod
    def load(cls, model_folder, name="sail_pipeline"):
        load_location = os.path.join(model_folder, name)

        # -------------------------------------------
        # Load steps to add to sail pipeline
        # -------------------------------------------
        steps = []
        steps_location = os.path.join(load_location, "steps")
        steps_meta = load_obj(
            location=steps_location,
            file_name="steps_meta",
            serialize_type="json",
        )
        for step_name in steps_meta:
            step_obj = load_obj(location=steps_location, file_name=step_name)
            steps.append((step_name, step_obj))

        # -------------------------------------------
        # Load params
        # -------------------------------------------
        params = load_obj(location=load_location, file_name="params")

        # -------------------------------------------
        # create pipeline
        # -------------------------------------------
        sail_pipeline = SAILPipeline(steps=steps, **params)

        # -------------------------------------------
        # Pre-load progressive scorer state from initial_points if present.
        # -------------------------------------------
        if os.path.exists(os.path.join(load_location, "scorer_state.npz")):
            initial_points = np.load(os.path.join(load_location, "scorer_state.npz"))
            sail_pipeline._scorer.progressive_score(
                initial_points["y_true"], initial_points["y_pred"], verbose=1
            )
            initial_points.close()

        return sail_pipeline
