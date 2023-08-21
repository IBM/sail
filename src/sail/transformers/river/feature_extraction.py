import river.feature_extraction.agg as agg
import river.feature_extraction.kernel_approx as kernel_approx
import river.feature_extraction.poly as poly
import river.feature_extraction.vectorize as vectorize
from river.compat import River2SKLTransformer
import typing
from river import stats

__all__ = [
    "Agg",
    "BagOfWords",
    "PolynomialExtender",
    "RBFSampler",
    "TargetAgg",
    "TFIDF",
]


class Agg(River2SKLTransformer):
    def __init__(
        self,
        on: str,
        by: typing.Optional[typing.Union[str, typing.List[str]]],
        how: stats.Univariate,
    ):
        super(Agg, self).__init__(
            river_estimator=agg.Agg(
                on,
                by,
                how,
            )
        )


class TargetAgg(River2SKLTransformer):
    def __init__(
        self,
        by: typing.Optional[typing.Union[str, typing.List[str]]],
        how: stats.Univariate,
        target_name="y",
    ):
        super(TargetAgg, self).__init__(
            river_estimator=agg.TargetAgg(
                by,
                how,
                target_name,
            )
        )


class RBFSampler(River2SKLTransformer):
    def __init__(self, gamma=1.0, n_components=100, seed: int = None):
        super(RBFSampler, self).__init__(
            river_estimator=kernel_approx.RBFSampler(gamma, n_components, seed)
        )


class PolynomialExtender(River2SKLTransformer):
    def __init__(
        self,
        degree=2,
        interaction_only=False,
        include_bias=False,
        bias_name="bias",
    ):
        super(PolynomialExtender, self).__init__(
            river_estimator=poly.PolynomialExtender(
                degree, interaction_only, include_bias, bias_name
            )
        )


class TFIDF(River2SKLTransformer):
    def __init__(
        self,
        normalize=True,
        on: str = None,
        strip_accents=True,
        lowercase=True,
        preprocessor: typing.Callable = None,
        tokenizer: typing.Callable = None,
        ngram_range=(1, 1),
    ):
        super(TFIDF, self).__init__(
            river_estimator=vectorize.TFIDF(
                normalize,
                on,
                strip_accents,
                lowercase,
                preprocessor,
                tokenizer,
                ngram_range,
            )
        )


class BagOfWords(River2SKLTransformer):
    def __init__(
        self,
        on: str = None,
        strip_accents=True,
        lowercase=True,
        preprocessor: typing.Callable = None,
        tokenizer: typing.Callable = None,
        ngram_range=(1, 1),
    ):
        super(BagOfWords, self).__init__(
            river_estimator=vectorize.BagOfWords(
                on,
                strip_accents,
                lowercase,
                preprocessor,
                tokenizer,
                ngram_range,
            )
        )
