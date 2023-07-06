import os
import shutil
import uuid
from typing import Any, List, Tuple
from sklearn import utils
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time

from sail.utils.logging import configure_logger
from sail.utils.progress_bar import SAILProgressBar
from sail.utils.scorer import SAILModelScorer
from sail.utils.serialization import load_obj, save_obj

LOGGER = configure_logger()


class SAILPipeline(Pipeline):
    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        scoring=None,
        verbose: int = 1,
    ):
        """[summary]

        Args:
            steps (list, optional): List of steps for the pipeline, required. Each step is a tuple containing a string, representing the name of a step, and a python object for the step.
            scoring: pipeline score metric
            verbosity: verbosity level for training logs. 0 (No logs) is default.
        """
        super(SAILPipeline, self).__init__(steps, verbose=False)
        self._uuid = uuid.uuid4()
        self.scoring = scoring
        self._scorer = SAILModelScorer(
            scoring=scoring, estimator=steps[-1][1], is_pipeline=True
        )
        self.log_verbose = verbose

    @property
    def progressive_score(self):
        return self._scorer.progressive_score

    def score(self, X, y, sample_weight=1.0):
        y_preds = self.predict(X)
        score = self._scorer.score(y_preds, y, sample_weight, verbose=self.log_verbose)
        return score

    def fit(self, X, y=None, **fit_params):
        self._fit(X, y, warm_start=False, **fit_params)

    def partial_fit(self, X, y=None, **fit_params):
        if self.__sklearn_is_fitted__():
            y_preds = self.predict(X)
            self._scorer._eval_progressive_score(y_preds, y, verbose=0)
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
            verbose=self.log_verbose,
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
                progress.update_params("P_Score", self.progressive_score)

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
                progress.update_params("P_Score", self.progressive_score)
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

    def save(self, model_folder, overwrite=True) -> str:
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
        save_location = os.path.join(model_folder, "sail_pipeline")
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
        # explicity save all the steps
        # -------------------------------------------
        for i, (name, step) in enumerate(self.steps):
            save_obj(
                obj=step, location=os.path.join(save_location, "steps"), file_name=name
            )

        # save the complete pipeline which internally calls __getstate__ to
        # remove unpickled objects
        save_obj(obj=self, location=save_location, file_name="pipeline")

        return save_location

    @classmethod
    def load(cls, model_folder):
        load_location = os.path.join(model_folder, "sail_pipeline")
        # load pipeline
        sail_pipeline = load_obj(location=load_location, file_name="pipeline")

        # load steps to pipeline
        for i, (name, _) in enumerate(sail_pipeline.steps):
            step_obj = load_obj(
                location=os.path.join(load_location, "steps"), file_name=name
            )

            sail_pipeline.steps[i] = (name, step_obj)

        return sail_pipeline
