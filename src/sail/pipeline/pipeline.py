import os
import shutil
from typing import Any, List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if

from sail.utils.logging import SAILVerbosity, configure_logger
from sail.utils.progress_bar import SAILProgressBar
from sail.utils.scorer import SAILModelScorer
from sail.utils.serialization import load_obj, save_obj

LOGGER = configure_logger(logger_name="SAILPipeline")


class SAILPipeline(Pipeline):
    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        scoring=None,
        cache_memory=None,
        verbosity: int | None | SAILVerbosity = 1,
    ):
        """[summary]

        Args:
            steps (list, optional): List of steps for the pipeline, required. Each step is a tuple containing a string, representing the name of a step, and a python object for the step.
            scoring: pipeline score metric
            verbosity: verbosity level for training logs. 0 (No logs) is default.
        """
        super(SAILPipeline, self).__init__(steps, memory=cache_memory, verbose=False)
        self.scoring = scoring
        self.cache_memory = cache_memory
        self.verbosity = self._resolve_verbosity(verbosity)
        self._scorer = self._validate_and_get_scorer(scoring, steps[-1][1])

    def _can_fit_transform(self):
        return self._final_estimator == "passthrough" or (
            hasattr(self._final_estimator, "fit")
            and hasattr(self._final_estimator, "transform")
        )

    def _can_parital_fit_transform(self):
        return self._final_estimator == "passthrough" or (
            hasattr(self._final_estimator, "partial_fit")
            and hasattr(self._final_estimator, "transform")
        )

    def _resolve_verbosity(self, verbosity):
        if verbosity is None:
            return SAILVerbosity(verbosity=0)
        elif isinstance(verbosity, int):
            if verbosity == 0 or verbosity == 1:
                return SAILVerbosity(verbosity=verbosity)
        elif isinstance(verbosity, SAILVerbosity):
            return verbosity
        else:
            raise Exception(
                "Invalid Verbosity value. Verbosity can only take values from [0, 1] or be an instance of the type SAILVerbosity"
            )

    def _validate_and_get_scorer(self, scoring, estimator):
        estimator_type = None

        if estimator == "passthrough":
            estimator_type = estimator
        elif hasattr(estimator, "_estimator_type"):
            estimator_type = estimator._estimator_type

        if estimator_type:
            estimator_type = SAILModelScorer(
                scoring=scoring,
                estimator_type=estimator_type,
                pipeline_mode=True,
            )

        return estimator_type

    @property
    def get_progressive_score(self):
        if self._scorer:
            return self._scorer.get_progressive_score

    def score(self, X, y, sample_weight=1.0, verbose: Literal[0, 1] | None = None):
        if self._scorer:
            y_preds = self.predict(X)
            if isinstance(y, pd.Series):
                y = y.to_numpy()
            return self._scorer.score(
                y, y_preds, sample_weight, verbose=self.verbosity.resolve(verbose)
            )

    def score_estimator(
        self, X, y, sample_weight: float = 1.0, verbose: Literal[0, 1] | None = None
    ):
        if self._scorer:
            y_preds = self._final_estimator.predict(X)
            if isinstance(y, pd.Series):
                y = y.to_numpy()
            return self._scorer.score(
                y, y_preds, sample_weight, verbose=self.verbosity.resolve(verbose)
            )

    def _progressive_score(
        self,
        X,
        y,
        sample_weight=1.0,
        detached=False,
        verbose: Literal[0, 1] | None = None,
    ):
        if self._scorer:
            y_pred = self.predict(X)
            if isinstance(y, pd.Series):
                y = y.to_numpy()
            return self._scorer.progressive_score(
                y, y_pred, sample_weight, detached, self.verbosity.resolve(verbose)
            )

    def fit(self, X, y=None, **fit_params):
        self.verbosity.log_epoch()
        self._fit(X, y, warm_start=False, **fit_params)

    def partial_fit(self, X, y=None, **fit_params):
        self.verbosity.log_epoch()
        self._partial_fit(X, y, **fit_params)

    def _partial_fit(self, X, y=None, **fit_params):
        if self.__sklearn_is_fitted__():
            self._progressive_score(X, y, verbose=0)
        self._fit(X, y, warm_start=True, **fit_params)

    def _fit(self, X, y=None, warm_start: bool = None, **fit_params):
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
            verbose=self.verbosity.get(),
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

            # call fit / partial_fit on the final estimator
            with _print_elapsed_time(
                "SAIL-FinalEstimator", self._log_message(len(self.steps) - 1)
            ):
                if self._final_estimator != "passthrough":
                    progress.append_desc(f"[{self.steps[-1][0]}]")
                    self.fit_final_estimator(
                        X, y, warm_start=warm_start, verbose=0, **fit_params_steps
                    )
                progress.update()

            # Update progress bar score after fit().
            if self.verbosity.get():
                if warm_start:
                    progress.update_params("P_Score", self.get_progressive_score)
                else:
                    progress.update_params(
                        "Score", self.score_estimator(X, y, verbose=0)
                    )

        return self

    def fit_final_estimator(
        self,
        X,
        y,
        warm_start: bool = False,
        verbose: Literal[0, 1] | None = None,
        **fit_params_steps,
    ):
        verbose = self.verbosity.resolve(verbose)
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

                # Update progress bar
                if verbose == 1:
                    progress.update()
                    progress.update_params("P_Score", self.get_progressive_score)
            else:
                if not hasattr(self._final_estimator, "fit"):
                    raise AttributeError(
                        f"Final Estimator '{self.steps[-1][0]}' does not implement fit()."
                    )
                fit_params_last_step.pop("classes", None)
                self._final_estimator.fit(X, y, **fit_params_last_step)

                # Update progress bar
                if verbose == 1:
                    progress.update()
                    progress.update_params(
                        "Score", self.score_estimator(X, y, verbose=0)
                    )

    @available_if(_can_fit_transform)
    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)

    @available_if(_can_parital_fit_transform)
    def partial_fit_transform(self, X, y=None, **fit_params):
        """Partial Fit the model and transform with the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        self.partial_fit(X, y, **fit_params)
        return self.transform(X)

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

            LOGGER.debug(f"making directory tree {save_location}")
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
