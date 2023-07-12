import importlib
import json
import os
import random
import time
import warnings
from operator import *
from typing import Dict, List

import numpy as np
from ray import tune
from ray.tune.experiment import Trial
from ray.tune.logger import LoggerCallback
from ray.tune.search import BasicVariantGenerator, ConcurrencyLimiter, Searcher
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.skopt import SkOptSearch
from ray.tune.sklearn import TuneGridSearchCV, TuneSearchCV
from ray.tune.stopper import CombinedStopper
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from tune_sklearn._trainable import _PipelineTrainable, _Trainable
from tune_sklearn.utils import (
    MaximumIterationStopper,
    check_is_pipeline,
    resolve_logger_callbacks,
)

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"

from sail.models.auto_ml.searcher import SailListSearcher, SailRandomListSearcher


def extract_top_best_configurations(
    num_of_configurations, metric_name, trial_results, params_grid
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
        configs = {param: configs[param] for param in eligible_keys if param in configs}
        best_configs.append(configs)

    return best_configs


class SAILTuneGridSearchCV(TuneGridSearchCV):
    default_search_params = {
        "verbose": 1,
        "max_iters": 1,
        "early_stopping": False,
        "mode": "max",
        "scoring": "accuracy",
        "pipeline_auto_early_stop": False,
        "keep_best_configurations": 1,
    }
    defined_loggers = ["csv", "json"]

    def __init__(
        self,
        num_cpus_per_trial=1,
        num_gpus_per_trial=0,
        keep_best_configurations=1,
        cluster_address=None,
        **kwargs,
    ):
        super(SAILTuneGridSearchCV, self).__init__(**kwargs)
        self.keep_best_configurations = keep_best_configurations
        self.cluster_address = cluster_address
        self.num_cpus_per_trial = num_cpus_per_trial
        self.num_gpus_per_trial = num_gpus_per_trial

    def fit(
        self, X, y=None, warm_start=False, groups=None, tune_params=None, **fit_params
    ):
        self.warm_start = warm_start
        return super().fit(X, y, groups, tune_params, **fit_params)

    def _list_grid_num_samples(self, warm_start):
        """Calculate the num_samples for `tune.run`.

        This is used when a list of dictionaries is passed in
        for the `param_grid`
        """
        num_samples = 0
        if (
            warm_start
            and hasattr(self, "_best_configurations")
            and self._best_configurations
        ):
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
            resources_per_trial={
                "cpu": self.num_cpus_per_trial,
                "gpu": self.num_gpus_per_trial,
            },
            trial_dirname_creator=lambda trial: f"Trail_{trial.trial_id}",
            name="SAILAutoML_Experiment" + "_" + time.strftime("%d-%m-%Y_%H:%M:%S"),
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
                    num_samples=self._list_grid_num_samples(self.warm_start),
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

        # Currently, there is a bug where a tqdm logger instance is left unhandled during the Ray Tune. Also, it is an overkill to log Pipeline Training for Distributed Cross Validation. Hence, turning off the verbosity for `SAILPipeline`. This does not affect the Ray Tune logs which can be set via verbose field of tune.run.
        for estimator in estimator_list:
            estimator.verbosity = 0

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
            self._best_configurations = extract_top_best_configurations(
                self.keep_best_configurations,
                self._metric_name,
                analysis.results,
                self.param_grid,
            )

        return analysis


class SAILTuneSearchCV(TuneSearchCV):
    default_search_params = {
        "verbose": 1,
        "scoring": "accuracy",
        "mode": "max",
        "early_stopping": False,
        "n_trials": 5,
        "search_optimization": "random",
        "pipeline_auto_early_stop": False,
        "keep_best_configurations": 1,
    }
    defined_loggers = ["csv", "json"]

    def __init__(
        self,
        num_cpus_per_trial=1,
        num_gpus_per_trial=0,
        keep_best_configurations=1,
        cluster_address=None,
        **kwargs,
    ):
        super(SAILTuneSearchCV, self).__init__(**kwargs)
        self.keep_best_configurations = keep_best_configurations
        self.cluster_address = cluster_address
        self.num_cpus_per_trial = num_cpus_per_trial
        self.num_gpus_per_trial = num_gpus_per_trial

    def fit(
        self, X, y=None, warm_start=False, groups=None, tune_params=None, **fit_params
    ):
        self.warm_start = warm_start
        return super().fit(X, y, groups, tune_params, **fit_params)

    def _tune_run(
        self, X, y, config, resources_per_trial, tune_params=None, fit_params=None
    ):
        """Wrapper to call ``tune.run``. Multiple estimators are generated when
        early stopping is possible, whereas a single estimator is
        generated when early stopping is not possible.

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
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        trainable = _Trainable
        if (
            self.pipeline_auto_early_stop
            and check_is_pipeline(self.estimator)
            and self.early_stopping_
        ):
            trainable = _PipelineTrainable

        max_iter = self.max_iters
        if self.early_stopping_ is not None:
            estimator_list = [clone(self.estimator) for _ in range(self.n_splits)]
            if hasattr(self.early_stopping_, "_max_t_attr"):
                # we want to delegate stopping to schedulers which
                # support it, but we want it to stop eventually, just in case
                # the solution is to make the stop condition very big
                max_iter = self.max_iters * 10
        else:
            estimator_list = [clone(self.estimator)]

        stopper = MaximumIterationStopper(max_iter=max_iter)
        if self.stopper:
            stopper = CombinedStopper(stopper, self.stopper)
        run_args = dict(
            scheduler=self.early_stopping_,
            reuse_actors=True,
            verbose=self.verbose,
            stop=stopper,
            num_samples=self.n_trials,
            config=config,
            fail_fast="raise",
            resources_per_trial={
                "cpu": self.num_cpus_per_trial,
                "gpu": self.num_gpus_per_trial,
            },
            local_dir=self.local_dir,
            name="SAILAutoML_Experiment" + "_" + time.strftime("%d-%m-%Y_%H:%M:%S"),
            callbacks=resolve_logger_callbacks(self.loggers, self.defined_loggers),
            time_budget_s=self.time_budget_s,
            metric=self._metric_name,
            mode=self.mode,
        )

        if self.warm_start and hasattr(self, "_best_configurations"):
            best_configurations = self._best_configurations
        else:
            best_configurations = None

        if self._search_optimization_lower == "random":
            if isinstance(self.param_distributions, list):
                search_algo = SailRandomListSearcher(
                    self.param_distributions, points_to_evaluate=best_configurations
                )
            else:
                search_algo = BasicVariantGenerator(
                    points_to_evaluate=best_configurations
                )
            run_args["search_alg"] = search_algo
        else:
            search_space = None
            override_search_space = True
            if (
                isinstance(self._search_optimization_lower, Searcher)
                and hasattr(self._search_optimization_lower, "_space")
                and self._search_optimization_lower._space is not None
            ):
                if self._search_optimization_lower._metric != self._metric_name:
                    raise ValueError(
                        "If a Searcher instance has been initialized with a "
                        "space, its metric "
                        f"('{self._search_optimization_lower._metric}') "
                        "must match the metric set in TuneSearchCV"
                        f" ('{self._metric_name}')"
                    )
                if self._search_optimization_lower._mode != self.mode:
                    raise ValueError(
                        "If a Searcher instance has been initialized with a "
                        "space, its mode "
                        f"('{self._search_optimization_lower._mode}') "
                        "must match the mode set in TuneSearchCV"
                        f" ('{self.mode}')"
                    )
            elif self._is_param_distributions_all_tune_domains():
                run_args["config"].update(self.param_distributions)
                override_search_space = False

            search_kwargs = self.search_kwargs or {}
            search_kwargs = search_kwargs.copy()
            if override_search_space:
                search_kwargs["metric"] = run_args.pop("metric")
                search_kwargs["mode"] = run_args.pop("mode")
                if run_args["scheduler"]:
                    if hasattr(run_args["scheduler"], "_metric") and hasattr(
                        run_args["scheduler"], "_mode"
                    ):
                        run_args["scheduler"]._metric = search_kwargs["metric"]
                        run_args["scheduler"]._mode = search_kwargs["mode"]
                    else:
                        warnings.warn(
                            "Could not set `_metric` and `_mode` attributes "
                            f"on Scheduler {run_args['scheduler']}. "
                            "This may cause an exception later! "
                            "Ensure your Scheduler initializes with those "
                            "attributes.",
                            UserWarning,
                        )

            if self._search_optimization_lower == "bayesian":
                if override_search_space:
                    search_space = self.param_distributions
                search_algo = SkOptSearch(
                    space=search_space,
                    points_to_evaluate=best_configurations,
                    **search_kwargs,
                )
                run_args["search_alg"] = search_algo

            elif self._search_optimization_lower == "bohb":
                if override_search_space:
                    search_space = self._get_bohb_config_space()
                search_algo = TuneBOHB(
                    space=search_space,
                    seed=self.seed,
                    points_to_evaluate=best_configurations,
                    **search_kwargs,
                )
                run_args["search_alg"] = search_algo

            elif self._search_optimization_lower == "optuna":
                if "sampler" not in search_kwargs:
                    module = importlib.import_module("optuna.samplers")
                    search_kwargs["sampler"] = getattr(module, "TPESampler")(
                        seed=self.seed
                    )
                elif self.seed:
                    warnings.warn("'seed' is not implemented for Optuna.")
                if override_search_space:
                    search_space = self._get_optuna_params()
                search_algo = OptunaSearch(
                    space=search_space,
                    points_to_evaluate=best_configurations,
                    **search_kwargs,
                )
                run_args["search_alg"] = search_algo

            elif self._search_optimization_lower == "hyperopt":
                if override_search_space:
                    search_space = self._get_hyperopt_params()
                search_algo = HyperOptSearch(
                    space=search_space,
                    random_state_seed=self.seed,
                    points_to_evaluate=best_configurations,
                    **search_kwargs,
                )
                run_args["search_alg"] = search_algo

            elif isinstance(self._search_optimization_lower, Searcher):
                search_algo = self._search_optimization_lower
                run_args["search_alg"] = search_algo

            else:
                # This should not happen as we validate the input before
                # this method. Still, just to be sure, raise an error here.
                raise ValueError(
                    "Invalid search optimizer: " f"{self._search_optimization_lower}"
                )

        if (
            isinstance(self.n_jobs, int)
            and self.n_jobs > 0
            and not self._searcher_name == "random"
        ):
            search_algo = ConcurrencyLimiter(search_algo, max_concurrent=self.n_jobs)
            run_args["search_alg"] = search_algo

        run_args = self._override_run_args_with_tune_params(run_args, tune_params)

        # Currently, there is a bug where a tqdm logger instance is left unhandled during the Ray Tune. Also, it is an overkill to log Pipeline Training for Distributed Cross Validation. Hence, turning off the verbosity for `SAILPipeline`. This does not affect the Ray Tune logs which can be set via verbose field of tune.run.
        for estimator in estimator_list:
            estimator.verbosity = 0

        trainable = tune.with_parameters(
            trainable, X=X, y=y, estimator_list=estimator_list, fit_params=fit_params
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="fail_fast='raise' " "detected.")
            analysis = tune.run(trainable, **run_args)

        if self.keep_best_configurations > 0:
            self._best_configurations = extract_top_best_configurations(
                self.keep_best_configurations,
                self._metric_name,
                analysis.results,
                self.param_distributions,
            )

        return analysis
