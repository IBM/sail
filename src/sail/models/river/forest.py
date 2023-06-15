import typing

import river.ensemble.adaptive_random_forest as adaptive_random_forest
from river import base, metrics
from river.drift import ADWIN
from river.tree.splitter import Splitter
from sail.models.river.base import SailRiverClassifier, SailRiverRegressor

__all__ = [
    "AdaptiveRandomForestClassifier",
    "AdaptiveRandomForestRegressor",
]


class AdaptiveRandomForestClassifier(SailRiverClassifier):
    def __init__(
        self,
        n_models: int = 10,
        max_features: typing.Union[bool, str, int] = "sqrt",
        lambda_value: int = 6,
        metric: metrics.base.MultiClassMetric = metrics.Accuracy(),
        disable_weighted_vote=False,
        drift_detector: typing.Union[base.DriftDetector, None] = ADWIN(delta=0.001),
        warning_detector: typing.Union[base.DriftDetector, None] = ADWIN(delta=0.01),
        # Tree parameters
        grace_period: int = 50,
        max_depth: int = None,
        split_criterion: str = "info_gain",
        delta: float = 0.01,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        binary_split: bool = False,
        max_size: float = 100.0,
        memory_estimate_period: int = 2_000_000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int = None,
    ):
        super(AdaptiveRandomForestClassifier, self).__init__(
            river_estimator=adaptive_random_forest.AdaptiveRandomForestClassifier(
                n_models,
                max_features,
                lambda_value,
                metric,
                disable_weighted_vote,
                drift_detector,
                warning_detector,
                grace_period,
                max_depth,
                split_criterion,
                delta,
                tau,
                leaf_prediction,
                nb_threshold,
                nominal_attributes,
                splitter,
                binary_split,
                max_size,
                memory_estimate_period,
                stop_mem_management,
                remove_poor_attrs,
                merit_preprune,
                seed,
            )
        )


class AdaptiveRandomForestRegressor(SailRiverRegressor):
    def __init__(
        self,
        n_models: int = 10,
        max_features="sqrt",
        aggregation_method: str = "median",
        lambda_value: int = 6,
        metric: metrics.base.RegressionMetric = metrics.MSE(),
        disable_weighted_vote=True,
        drift_detector: base.DriftDetector = ADWIN(0.001),
        warning_detector: base.DriftDetector = ADWIN(0.01),
        # Tree parameters
        grace_period: int = 50,
        max_depth: int = None,
        delta: float = 0.01,
        tau: float = 0.05,
        leaf_prediction: str = "adaptive",
        leaf_model: base.Regressor = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: float = 500.0,
        memory_estimate_period: int = 2_000_000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int = None,
    ):
        super(AdaptiveRandomForestRegressor, self).__init__(
            river_estimator=adaptive_random_forest.AdaptiveRandomForestRegressor(
                n_models,
                max_features,
                aggregation_method,
                lambda_value,
                metric,
                disable_weighted_vote,
                drift_detector,
                warning_detector,
                grace_period,
                max_depth,
                delta,
                tau,
                leaf_prediction,
                leaf_model,
                model_selector_decay,
                nominal_attributes,
                splitter,
                min_samples_split,
                binary_split,
                max_size,
                memory_estimate_period,
                stop_mem_management,
                remove_poor_attrs,
                merit_preprune,
                seed,
            )
        )
