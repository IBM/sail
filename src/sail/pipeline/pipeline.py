import uuid
from typing import List, Tuple

from sklearn import utils
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time

from sail.utils.logging import configure_logger

LOGGER = configure_logger()


class SAILPipeline(Pipeline):
    def __init__(self, steps: List[Tuple] = [], verbose=True):
        """[summary]

        Args:
            steps (list, optional): List of steps for the pipeline, required. Each step is a tuple containing a string, representing the name of a step, and a python object for the step.
        """

        super(SAILPipeline, self).__init__(steps, verbose=verbose)
        self._uuid = uuid.uuid4()

        LOGGER.info("created SAILPipeline object with ID %s", self._uuid)

    def fit(self, X, y=None, **fit_params):
        self._partial_fit(X, y, **fit_params)

    def partial_fit(self, X, y=None, **fit_params):
        LOGGER.info("Calling Partial_fit() on the pipeline.")
        self._partial_fit(X, y, **fit_params)

    def _partial_fit(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = utils.validation.check_memory(self.memory)

        def _fit_transform_one(transformer, X, y, **fit_params):
            if hasattr(transformer, "partial_fit"):
                transformed_X = transformer.partial_fit(X, y, **fit_params).transform(X)
            elif hasattr(transformer, "fit_transform"):
                transformed_X = transformer.fit_transform(X, y, **fit_params)
            else:
                transformed_X = transformer.fit(X, y, **fit_params).transform(X)
            return transformed_X, transformer

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=True
        ):
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

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                if not hasattr(self._final_estimator, "partial_fit"):
                    raise AttributeError(
                        f"Final Estimator '{self.steps[-1][0]}' does not implement partial_fit()."
                    )
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                if (
                    self._final_estimator._estimator_type == "classifier"
                    and "classes" not in fit_params_last_step
                ):
                    fit_params_last_step["classes"] = utils.multiclass.unique_labels(y)
                self._final_estimator.partial_fit(X, y, **fit_params_last_step)

        return self
