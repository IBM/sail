from .base import SAILTransformer
import pandas as pd
from sklearn.base import ClassNamePrefixFeaturesOutMixin
from sklearn.utils.validation import check_is_fitted


class EncodeDateTransformer(ClassNamePrefixFeaturesOutMixin, SAILTransformer):
    def __init__(self, datetime_col, temporal_fields, prefix=None) -> None:
        super(EncodeDateTransformer, self).__init__()
        self.datetime_col = datetime_col
        self.temporal_fields = temporal_fields
        self.prefix = prefix
        self.col = (prefix if prefix else datetime_col) + "_"
        self.validation_params["cast_to_ndarray"] = False

    def _partial_fit(self, X, y=None):
        return self

    def _transform(self, X, y=None, copy=None):
        new_df = pd.DataFrame()

        if self.datetime_col not in X.columns:
            raise Exception(f"Datetime column: {self.datetime_col} not available in X.")

        for field in self.temporal_fields:
            if "time" == field:
                new_df[self.col + field] = (
                    X[self.datetime_col].dt.hour + X[self.datetime_col].dt.minute / 60.0
                )

            if "day" == field:
                new_df[self.col + field] = X[self.datetime_col].dt.dayofweek

            if "month" == field:
                new_df[self.col + field] = X[self.datetime_col].dt.month - 1

            if "is_weekend" == field:
                new_df[self.col + field] = (
                    X[self.datetime_col].dt.dayofweek >= 5
                ).astype("int")

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
