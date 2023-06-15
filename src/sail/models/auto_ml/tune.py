import json
import os
import warnings
from glob import glob
from operator import *
from typing import Dict, List

import ray.cloudpickle as cpickle
from ray import tune
from ray.tune.experiment import Trial
from ray.tune.logger import LoggerCallback
from ray.tune.search import BasicVariantGenerator
from ray.tune.sklearn import TuneGridSearchCV, TuneSearchCV
from ray.tune.stopper import CombinedStopper
from ray.tune.utils.util import SafeFallbackEncoder
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from tune_sklearn._trainable import _PipelineTrainable, _Trainable
from tune_sklearn.utils import (
    EarlyStopping,
    MaximumIterationStopper,
    check_is_pipeline,
    resolve_logger_callbacks,
)

from sail.models.auto_ml.searcher import SailListSearcher


class SAILTuneGridSearchCV(TuneGridSearchCV):
    def __init__(self, *args, keep_best_configurations=1, **kwargs):
        super(SAILTuneGridSearchCV, self).__init__(*args, **kwargs)
        self.keep_best_configurations = keep_best_configurations

    defined_loggers = ["csv", "mlflow", "json"]

    def fit(
        self, X, y=None, warm_start=False, groups=None, tune_params=None, **fit_params
    ):
        self.warm_start = warm_start
        return super().fit(X, y, groups, tune_params, **fit_params)

    def _list_grid_num_samples(self):
        """Calculate the num_samples for `tune.run`.

        This is used when a list of dictionaries is passed in
        for the `param_grid`
        """
        num_samples = 0
        if hasattr(self, "_best_configurations") and self._best_configurations:
            num_samples = len(self._best_configurations)
        return num_samples + len(list(ParameterGrid(self.param_grid)))

    def _num_samples(self, list_searcher=False):
        """Calculate the num_samples for `tune.run`.

        This is used when a dictionaries is passed in
        for the `param_grid`
        """
        num_samples = 0
        if hasattr(self, "_best_configurations") and self._best_configurations:
            num_samples = len(self._best_configurations)
        return num_samples + 1

    def _tune_run(
        self, X, y, config, resources_per_trial, tune_params=None, fit_params=None
    ):
        """Wrapper to call ``tune.run``. Multiple estimators are generated when
        early stopping is possible, whereas a single estimator is
        generated when  early stopping is not possible.

        Args:
            X (:obj:`array-like` (shape = [n_samples, n_features])):
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            y (:obj:`array-like`): Shape of array expected to be [n_samples]
                or [n_samples, n_output]). Target relative to X for
                classification or regression; None for unsupervised learning.
            config (dict): Configurations such as hyperparameters to run
                ``tune.run`` on.
            resources_per_trial (dict): Resources to use per trial within Ray.
                Accepted keys are `cpu`, `gpu` and custom resources, and values
                are integers specifying the number of each resource to use.
            tune_params (dict): User defined parameters passed to
                ``tune.run``. Parameters inside `tune_params` override
                preset parameters.
            fit_params (dict): Parameters passed to the ``fit`` method
                of the estimator.

        Returns:
            analysis (`ExperimentAnalysis`): Object returned by
                `tune.run`.

        """
        trainable = _Trainable
        if (
            self.pipeline_auto_early_stop
            and check_is_pipeline(self.estimator)
            and self.early_stopping_
        ):
            trainable = _PipelineTrainable

        if self.early_stopping_ is not None:
            estimator_list = [clone(self.estimator) for _ in range(self.n_splits)]
        else:
            estimator_list = [clone(self.estimator)]

        stopper = MaximumIterationStopper(max_iter=self.max_iters)
        if self.stopper:
            stopper = CombinedStopper(stopper, self.stopper)

        run_args = dict(
            scheduler=self.early_stopping_,
            reuse_actors=True,
            verbose=self.verbose,
            stop=stopper,
            config=config,
            fail_fast="raise",
            resources_per_trial=resources_per_trial,
            local_dir=self.local_dir,
            name=self.name,
            callbacks=resolve_logger_callbacks(self.loggers, self.defined_loggers),
            time_budget_s=self.time_budget_s,
            metric=self._metric_name,
            mode=self.mode,
        )

        if self.warm_start and hasattr(self, "_best_configurations"):
            best_configurations = self._best_configurations
        else:
            best_configurations = None

        if isinstance(self.param_grid, list):
            run_args.update(
                dict(
                    search_alg=SailListSearcher(
                        self.param_grid, points_to_evaluate=best_configurations
                    ),
                    num_samples=self._list_grid_num_samples(),
                )
            )
        else:
            max_concurrent_trials = tune_params.get("max_concurrent_trials", 0)
            run_args.update(
                dict(
                    search_alg=BasicVariantGenerator(
                        max_concurrent=max_concurrent_trials,
                        points_to_evaluate=best_configurations,
                    ),
                    num_samples=self._num_samples(),
                )
            )

        run_args = self._override_run_args_with_tune_params(run_args, tune_params)

        trainable = tune.with_parameters(
            trainable,
            X=X,
            y=y,
            estimator_list=estimator_list,
            fit_params=fit_params,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="fail_fast='raise' " "detected.")
            analysis = tune.run(
                trainable,
                **run_args,
            )

        if self.keep_best_configurations > 0:
            self._best_configurations = self.extract_top_best_configurations(
                self.keep_best_configurations,
                self._metric_name,
                analysis.results,
                self.param_grid,
            )

        return analysis

    def extract_top_best_configurations(
        self, num_of_configurations, metric_name, trial_results, params_grid
    ):
        best_trials = sorted(
            trial_results.items(),
            reverse=True,
            key=lambda trial: getitem(trial[1], metric_name),
        )[0:num_of_configurations]

        eligible_keys = []
        if isinstance(params_grid, list):
            for params in params_grid:
                for param, _ in params.items():
                    eligible_keys.append(param)
        else:
            for param, _ in params_grid.items():
                eligible_keys.append(param)

        best_configs = []
        for trial, trial_results in best_trials:
            configs = trial_results["config"]
            configs = {
                param: configs[param] for param in eligible_keys if param in configs
            }
            best_configs.append(configs)

        return best_configs


class SAILTuneSearchCV(TuneSearchCV):
    ...
