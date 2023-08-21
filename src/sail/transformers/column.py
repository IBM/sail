from sklearn.base import BaseEstimator, TransformerMixin


class ColumnNamePrefixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, prefix="") -> None:
        self.prefix = prefix

    def fit(self, X: list, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        for feature in X.columns:
            X[self.prefix + "_" + feature] = X[feature]

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
