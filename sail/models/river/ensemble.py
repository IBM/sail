from river import base, metrics
from river.drift import ADWIN
from river.metrics.base import Metric
from river.tree.splitter import Splitter
import river.ensemble.bagging as bagging
import river.ensemble.boosting as boosting
import river.ensemble.streaming_random_patches as streaming_random_patches
import river.ensemble.adaptive_random_forest as adaptive_random_forest
from river.compat import River2SKLRegressor, River2SKLClassifier
import typing


__all__ = [
    "AdaptiveRandomForestClassifier",
    "AdaptiveRandomForestRegressor",
    "AdaBoostClassifier",
    "ADWINBaggingClassifier",
    "BaggingClassifier",
    "BaggingRegressor",
    "LeveragingBaggingClassifier",
    "SRPClassifier",
]


class BaggingClassifier(River2SKLClassifier):
    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super(BaggingClassifier, self).__init__(
            river_estimator=bagging.BaggingClassifier(model, n_models, seed)
        )


class BaggingRegressor(River2SKLRegressor):
    def __init__(self, model: base.Regressor, n_models=10, seed: int = None):
        super(BaggingRegressor, self).__init__(
            river_estimator=bagging.BaggingRegressor(model, n_models, seed)
        )


class ADWINBaggingClassifier(River2SKLClassifier):
    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super(ADWINBaggingClassifier, self).__init__(
            river_estimator=bagging.ADWINBaggingClassifier(model, n_models, seed)
        )


class LeveragingBaggingClassifier(River2SKLClassifier):
    def __init__(
        self,
        model: base.Classifier,
        n_models: int = 10,
        w: float = 6,
        adwin_delta: float = 0.002,
        bagging_method: str = "bag",
        seed: int = None,
    ):
        super(LeveragingBaggingClassifier, self).__init__(
            river_estimator=bagging.LeveragingBaggingClassifier(
                model, n_models, w, adwin_delta, bagging_method, seed
            )
        )


class AdaBoostClassifier(River2SKLClassifier):
    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super(AdaBoostClassifier, self).__init__(
            river_estimator=boosting.AdaBoostClassifier(model, n_models, seed)
        )


class AdaptiveRandomForestClassifier(River2SKLClassifier):
    def __init__(
        self,
        n_models: int = 10,
        max_features: typing.Union[bool, str, int] = "sqrt",
        lambda_value: int = 6,
        metric: metrics.MultiClassMetric = metrics.Accuracy(),
        disable_weighted_vote=False,
        drift_detector: typing.Union[base.DriftDetector, None] = ADWIN(delta=0.001),
        warning_detector: typing.Union[base.DriftDetector, None] = ADWIN(delta=0.01),
        grace_period: int = 50,
        max_depth: int = None,
        split_criterion: str = "info_gain",
        split_confidence: float = 0.01,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        binary_split: bool = False,
        max_size: int = 32,
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
                split_confidence,
                tie_threshold,
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


class AdaptiveRandomForestRegressor(River2SKLRegressor):
    def __init__(
        self,
        n_models: int = 10,
        max_features="sqrt",
        aggregation_method: str = "median",
        lambda_value: int = 6,
        metric: metrics.RegressionMetric = metrics.MSE(),
        disable_weighted_vote=True,
        drift_detector: base.DriftDetector = ADWIN(0.001),
        warning_detector: base.DriftDetector = ADWIN(0.01),
        grace_period: int = 50,
        max_depth: int = None,
        split_confidence: float = 0.01,
        tie_threshold: float = 0.05,
        leaf_prediction: str = "model",
        leaf_model: base.Regressor = None,
        model_selector_decay: float = 0.95,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        min_samples_split: int = 5,
        binary_split: bool = False,
        max_size: int = 500,
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
                split_confidence,
                tie_threshold,
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


class SRPClassifier(River2SKLClassifier):
    def __init__(
        self,
        model: base.Estimator = None,
        n_models: int = 10,
        subspace_size: typing.Union[int, float, str] = 0.6,
        training_method: str = "patches",
        lam: float = 6.0,
        drift_detector: base.DriftDetector = None,
        warning_detector: base.DriftDetector = None,
        disable_detector: str = "off",
        disable_weighted_vote: bool = False,
        seed=None,
        metric: Metric = None,
    ):
        super(SRPClassifier, self).__init__(
            river_estimator=streaming_random_patches.SRPClassifier(
                model,
                n_models,
                subspace_size,
                training_method,
                lam,
                drift_detector,
                warning_detector,
                disable_detector,
                disable_weighted_vote,
                seed,
                metric,
            )
        )
