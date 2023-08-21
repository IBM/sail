from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Polar2CartTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n, features=None) -> None:
        self.n = n
        self.features = features

    def fit(self, X: list, y=None):
        return self

    def transform(self, X, y=None):
        features = self.features if self.features else X.columns
        for feature in features:
            name_x = feature + "_x"
            X[name_x] = np.cos(2 * np.pi * X[feature] / self.n)
            name_y = feature + "_y"
            X[name_y] = np.sin(2 * np.pi * X[feature] / self.n)

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
