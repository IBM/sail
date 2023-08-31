from .base import SAILTransformer
import numpy as np
import pandas as pd
from sklearn.base import ClassNamePrefixFeaturesOutMixin
from sklearn.utils.validation import check_is_fitted


class Polar2CartTransformer(ClassNamePrefixFeaturesOutMixin, SAILTransformer):
    def __init__(self, n, features=None, suffix_x="x", suffix_y="y") -> None:
        super(Polar2CartTransformer, self).__init__()
        self.n = n
        self.features = features
        self.suffix_x = suffix_x
        self.suffix_y = suffix_y
        self.validation_params["cast_to_ndarray"] = False

    def _partial_fit(self, X, y=None):
        return self

    def _transform(self, X, y=None, copy=None):
        new_df = pd.DataFrame()

        features = self.features if self.features else X.columns
        for feature in features:
            name_x = feature + "_" + self.suffix_x
            new_df[name_x] = np.cos(2 * np.pi * X[feature] / self.n)
            name_y = feature + "_" + self.suffix_y
            new_df[name_y] = np.sin(2 * np.pi * X[feature] / self.n)

        self.features_ = list(new_df.columns)
        return new_df.to_numpy()

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
        return self.features_
