from river import base, compose
from sail.transformers.river.base import BaseRiverTransformer
from sklearn.base import ClassNamePrefixFeaturesOutMixin
from sklearn.utils.validation import check_is_fitted


class Select(ClassNamePrefixFeaturesOutMixin, BaseRiverTransformer):
    def __init__(self, keys: List[base.typing.FeatureName]):
        super(Select, self).__init__(river_estimator=compose.Select(*keys))
        self.validation_params["cast_to_ndarray"] = False

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Same as input features.
        """
        check_is_fitted(self, "n_features_in_")
        return list(self.instance_.keys)
