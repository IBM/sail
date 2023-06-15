import typing

import river.ensemble.bagging as bagging
import river.ensemble.boosting as boosting
import river.ensemble.streaming_random_patches as streaming_random_patches
from river import base
from river.metrics.base import Metric

from sail.models.river.base import SailRiverClassifier, SailRiverRegressor

__all__ = [
    "AdaBoostClassifier",
    "ADWINBaggingClassifier",
    "BaggingClassifier",
    "BaggingRegressor",
    "LeveragingBaggingClassifier",
    "SRPClassifier",
]


class BaggingClassifier(SailRiverClassifier):
    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super(BaggingClassifier, self).__init__(
            river_estimator=bagging.BaggingClassifier(model, n_models, seed)
        )


class BaggingRegressor(SailRiverRegressor):
    def __init__(self, model: base.Regressor, n_models=10, seed: int = None):
        super(BaggingRegressor, self).__init__(
            river_estimator=bagging.BaggingRegressor(model, n_models, seed)
        )


class ADWINBaggingClassifier(SailRiverClassifier):
    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super(ADWINBaggingClassifier, self).__init__(
            river_estimator=bagging.ADWINBaggingClassifier(model, n_models, seed)
        )


class LeveragingBaggingClassifier(SailRiverClassifier):
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


class AdaBoostClassifier(SailRiverClassifier):
    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super(AdaBoostClassifier, self).__init__(
            river_estimator=boosting.AdaBoostClassifier(model, n_models, seed)
        )


class SRPClassifier(SailRiverClassifier):
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
