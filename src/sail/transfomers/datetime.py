from sklearn.base import BaseEstimator, TransformerMixin


class EncodeDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col, temporal_fields, prefix="datetime") -> None:
        self.prefix = prefix
        self.datetime_col = datetime_col
        self.temporal_fields = temporal_fields
        self.col = prefix + "_"

    def fit(self, X: list, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        if self.datetime_col not in X.columns:
            raise Exception(f"Datetime column: {self.datetime_col} not available in X.")

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
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def set_output(self, *, transform=None):
        """Set output container.

        See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
        for an example on how to use the API.

        Parameters
        ----------
        transform : {"default", "pandas"}, default=None
            Configure output of `transform` and `fit_transform`.

            - `"default"`: Default output format of a transformer
            - `"pandas"`: DataFrame output
            - `None`: Transform configuration is unchanged

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if hasattr(super(), "set_output"):
            return super().set_output(transform=transform)

        return self
