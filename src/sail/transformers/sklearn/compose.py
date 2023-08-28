from sklearn import compose
import numpy as np
from scipy import sparse
from sklearn.utils import _print_elapsed_time


def _partial_fit_transform_one(
    transformer, X, y, weight, message_clsname="", message=None, **fit_params
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, "partial_fit_transform"):
            res = transformer.partial_fit_transform(X, y, **fit_params)
        elif hasattr(transformer, "partial_fit"):
            res = transformer.partial_fit(X, y, **fit_params).transform(X)
        else:
            res = transformer.fit(X, y, **fit_params).transform(X)

    if weight is None:
        return res, transformer
    return res * weight, transformer


class ColumnTransformer(compose.ColumnTransformer):
    def partial_fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.

        Returns
        -------
        self : ColumnTransformer
            This estimator.
        """
        # we use fit_transform to make sure to set sparse_output_ (for which we
        # need the transformed data) to have consistent output type in predict
        self.partial_fit_transform(X, y=y)
        return self

    def partial_fit_transform(self, X, y=None):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples,), default=None
            Targets for supervised learning.

        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            Horizontally stacked results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        self._check_feature_names(X, reset=True)

        X = compose._column_transformer._check_X(X)
        # set n_features_in_ attribute
        self._check_n_features(X, reset=True)
        self._validate_transformers()
        self._validate_column_callables(X)
        self._validate_remainder(X)

        result = self._fit_transform(X, y, _partial_fit_transform_one)

        if not result:
            self._update_fitted_transformers([])
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)

        # determine if concatenated output will be sparse or not
        if any(sparse.issparse(X) for X in Xs):
            nnz = sum(X.nnz if sparse.issparse(X) else X.size for X in Xs)
            total = sum(
                X.shape[0] * X.shape[1] if sparse.issparse(X) else X.size for X in Xs
            )
            density = nnz / total
            self.sparse_output_ = density < self.sparse_threshold
        else:
            self.sparse_output_ = False

        self._update_fitted_transformers(transformers)
        self._validate_output(Xs)
        self._record_output_indices(Xs)

        return self._hstack(list(Xs))
