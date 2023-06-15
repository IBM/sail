import river.feature_selection.k_best as k_best
import river.feature_selection.random as random
import river.feature_selection.variance as variance
from river.compat import River2SKLTransformer
from river import stats

__all__ = ["PoissonInclusion", "SelectKBest", "VarianceThreshold"]


class SelectKBest(River2SKLTransformer):
    def __init__(self, similarity: stats.Bivariate, k=10):
        super(SelectKBest, self).__init__(
            river_estimator=k_best.SelectKBest(similarity, k)
        )


class PoissonInclusion(River2SKLTransformer):
    def __init__(self, p: float, seed: int = None):
        super(PoissonInclusion, self).__init__(
            river_estimator=random.PoissonInclusion(p, seed)
        )


class VarianceThreshold(River2SKLTransformer):
    def __init__(self, threshold=0, min_samples=2):
        super(VarianceThreshold, self).__init__(
            river_estimator=variance.VarianceThreshold(threshold, min_samples)
        )
