import river.preprocessing.feature_hasher as feature_hasher
import river.preprocessing.impute as impute
import river.preprocessing.lda as lda
import river.preprocessing.one_hot as one_hot
import river.preprocessing.scale as scale
from sail.transfomers.river.base import BaseRiverTransformer

__all__ = [
    "AdaptiveStandardScaler",
    "Binarizer",
    "FeatureHasher",
    "LDA",
    "MaxAbsScaler",
    "MinMaxScaler",
    "Normalizer",
    "OneHotEncoder",
    "PreviousImputer",
    "RobustScaler",
    "StandardScaler",
    "StatImputer",
]


class FeatureHasher(BaseRiverTransformer):
    def __init__(self, n_features=1048576, seed: int = None):
        super(FeatureHasher, self).__init__(
            river_estimator=feature_hasher.FeatureHasher(n_features, seed)
        )


class PreviousImputer(BaseRiverTransformer):
    def __init__(self):
        super(PreviousImputer, self).__init__(
            river_estimator=impute.PreviousImputer()
        )


class StatImputer(BaseRiverTransformer):
    def __init__(self, *imputers):
        super(StatImputer, self).__init__(
            river_estimator=impute.StatImputer(*imputers)
        )


class LDA(BaseRiverTransformer):
    def __init__(
        self,
        n_components=10,
        number_of_documents=1e6,
        alpha_theta=0.5,
        alpha_beta=100.0,
        tau=64.0,
        kappa=0.75,
        vocab_prune_interval=10,
        number_of_samples=10,
        ranking_smooth_factor=1e-12,
        burn_in_sweeps=5,
        maximum_size_vocabulary=4000,
        seed: int = None,
    ):
        super(LDA, self).__init__(
            river_estimator=lda.LDA(
                n_components,
                number_of_documents,
                alpha_theta,
                alpha_beta,
                tau,
                kappa,
                vocab_prune_interval,
                number_of_samples,
                ranking_smooth_factor,
                burn_in_sweeps,
                maximum_size_vocabulary,
                seed,
            )
        )


class OneHotEncoder(BaseRiverTransformer):
    def __init__(self, sparse=False):
        super(OneHotEncoder, self).__init__(
            river_estimator=one_hot.OneHotEncoder(sparse)
        )


class AdaptiveStandardScaler(BaseRiverTransformer):
    def __init__(self, alpha=0.3):
        super(AdaptiveStandardScaler, self).__init__(
            river_estimator=scale.AdaptiveStandardScaler(alpha)
        )


class Binarizer(BaseRiverTransformer):
    def __init__(self, threshold=0.0, dtype=bool):
        super(Binarizer, self).__init__(
            river_estimator=scale.Binarizer(threshold, dtype)
        )


class MaxAbsScaler(BaseRiverTransformer):
    def __init__(self):
        super(MaxAbsScaler, self).__init__(
            river_estimator=scale.MaxAbsScaler()
        )


class MinMaxScaler(BaseRiverTransformer):
    def __init__(self):
        super(MinMaxScaler, self).__init__(
            river_estimator=scale.MinMaxScaler()
        )


class Normalizer(BaseRiverTransformer):
    def __init__(self, order=2):
        super(Normalizer, self).__init__(
            river_estimator=scale.Normalizer(order)
        )


class RobustScaler(BaseRiverTransformer):
    def __init__(
        self, with_centering=True, with_scaling=True, q_inf=0.25, q_sup=0.75
    ):
        super(RobustScaler, self).__init__(
            river_estimator=scale.RobustScaler(
                with_centering, with_scaling, q_inf, q_sup
            )
        )


class StandardScaler(BaseRiverTransformer):
    def __init__(self, with_std=True):
        super(StandardScaler, self).__init__(
            river_estimator=scale.StandardScaler(with_std)
        )
