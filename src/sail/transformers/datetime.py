from .base import SAILTransformer
import pandas as pd
from sklearn.base import OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted


class EncodeDateTransformer(OneToOneFeatureMixin, SAILTransformer):
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
        if self.datetime_col not in X.columns:
            raise Exception(f"Datetime column: {self.datetime_col} not available in X.")

        X = X.copy()
        for field in self.temporal_fields:
            if "time" == field:
                X[self.col + field] = (
                    X[self.datetime_col].dt.hour + X[self.datetime_col].dt.minute / 60.0
                )

            if "day" == field:
                X[self.col + field] = X[self.datetime_col].dt.dayofweek

            if "month" == field:
                X[self.col + field] = X[self.datetime_col].dt.month - 1

            if "is_weekend" == field:
                X[self.col + field] = (X[self.datetime_col].dt.dayofweek >= 5).astype(
                    "int"
                )

        self.feature_names_in_ = list(X.columns)
        return X.to_numpy()
