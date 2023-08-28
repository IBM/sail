import river.feature_extraction.agg as agg
import river.feature_extraction.kernel_approx as kernel_approx
import river.feature_extraction.poly as poly
import river.feature_extraction.vectorize as vectorize
from sail.transformers.river.base import BaseRiverTransformer
import typing
from river import stats, utils
from sklearn.base import ClassNamePrefixFeaturesOutMixin
import numpy as np

__all__ = [
    "Agg",
    "BagOfWords",
    "PolynomialExtender",
    "RBFSampler",
    "TargetAgg",
    "TFIDF",
]


class Agg(BaseRiverTransformer):
    def __init__(
        self,
        on: str,
        by: str | list[str] | None,
        how: stats.base.Univariate | utils.Rolling | utils.TimeRolling,
    ):
        super(Agg, self).__init__(
            river_estimator=agg.Agg(
                on,
                by,
                how,
            )
        )


class TargetAgg(ClassNamePrefixFeaturesOutMixin, BaseRiverTransformer):
    def __init__(
        self,
        by: str | list[str] | None,
        how: stats.base.Univariate | utils.Rolling | utils.TimeRolling,
        target_name="y",
    ):
        super(TargetAgg, self).__init__(
            river_estimator=agg.TargetAgg(
                by,
                how,
                target_name,
            )
        )
        self.validation_params["cast_to_ndarray"] = False

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        The feature names out will prefixed by the lowercased class name. For
        example, if the transformer outputs 3 features, then the feature names
        out are: `["class_name0", "class_name1", "class_name2"]`.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Only used to validate feature names with the names seen in :meth:`fit`.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        self._feature_name = f"{self.on}_{self.how.name}"
        if self.by:
            self._feature_name += f"_by_{'_and_'.join(self.by)}"
        return np.asarray([self._feature_name], dtype=object)


class RBFSampler(BaseRiverTransformer):
    def __init__(self, gamma=1.0, n_components=100, seed: int | None = None):
        super(RBFSampler, self).__init__(
            river_estimator=kernel_approx.RBFSampler(gamma, n_components, seed)
        )


class PolynomialExtender(BaseRiverTransformer):
    def __init__(
        self, degree=2, interaction_only=False, include_bias=False, bias_name="bias"
    ):
        super(PolynomialExtender, self).__init__(
            river_estimator=poly.PolynomialExtender(
                degree, interaction_only, include_bias, bias_name
            )
        )


class TFIDF(BaseRiverTransformer):
    def __init__(
        self,
        normalize=True,
        on: str | None = None,
        strip_accents=True,
        lowercase=True,
        preprocessor: typing.Callable | None = None,
        tokenizer: typing.Callable | None = None,
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


class BagOfWords(BaseRiverTransformer):
    def __init__(
        self,
        on: str | None = None,
        strip_accents=True,
        lowercase=True,
        preprocessor: typing.Callable | None = None,
        stop_words: set[str] | None = None,
        tokenizer_pattern=r"(?u)\b\w[\w\-]+\b",
        tokenizer: typing.Callable | None = None,
        ngram_range=(1, 1),
    ):
        super(BagOfWords, self).__init__(
            river_estimator=vectorize.BagOfWords(
                on,
                strip_accents,
                lowercase,
                preprocessor,
                stop_words,
                tokenizer_pattern,
                tokenizer,
                ngram_range,
            )
        )
