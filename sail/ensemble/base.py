"""
Base class for ensemble models
Based on: https://github.com/yzhao062/combo/blob/master/combo/models/base.py
and extended for distributed incremental models
"""

import warnings
from collections import defaultdict
from abc import ABC, abstractmethod
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets
import numpy as np
from inspect import signature


class BaseAggregator(ABC):

    @abstractmethod
    def __init__(self, base_estimators, fitted_estimators=[], window_size=200, pre_fitted=False):
        """
        :param base_estimators: Estimator objects with partial_fit defined
                    (incremental learning models).
        :param fitted_estimators: Batch learning estimators.
                Used only for scoring.
        :param metrics: Object that defines the quality of predictions
           (ex. metrics.accuracy_score in scikit-learn)
        """
        assert (isinstance(base_estimators, (list)))
        assert (isinstance(fitted_estimators, (list)))
        if (len(base_estimators) + len(fitted_estimators)) < 2:
            raise ValueError('At least 2 estimators are required')
        self.base_estimators = base_estimators
        self.fitted_estimators = fitted_estimators
        self.n_base_estimators_ = len(self.base_estimators)
        self.n_fitted_estimators_ = len(self.fitted_estimators)
        self.window_size = window_size

    def fit(self, X, y=None):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :return:
        """
        return self.partial_fit(X,y)

    @abstractmethod
    def partial_fit(self, X, y=None, classes=None):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :param y: numpy.ndarray of shape (n_samples)
               Labels for the target variable.
        :param classes: numpy.ndarray, optional.
               Unique classes in the data y.
        :return:
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
        :return: numpy array of shape (n_samples,).
             Class labels/predictions for input samples.
        """
        pass

    # @abstractmethod
    # def predict_proba(self, X):
    #     """Return probability estimates for the test data X.
    #     Parameters
    #     ----------
    #     X : numpy array of shape (n_samples, n_features)
    #         The input samples.
    #     Returns
    #     -------
    #     p : numpy array of shape (n_samples,)
    #         The class probabilities of the input samples.
    #         Classes are ordered by lexicographic order.
    #     """
    #     pass

    def _set_n_classes(self, y):
        self._classes = 2  # default as binary classification
        if y is not None:
            check_classification_targets(y)
            self._classes = len(np.unique(y))

        return self

    def _set_weights(self, weights):
        """Internal function to set estimator weights.
        Parameters
        ----------
        weights : numpy array of shape (n_estimators,)
            Estimator weights. May be used after the alignment.
        Returns
        -------
        self
        """

        if weights is None:
            self.weights = np.ones([1, self.n_base_estimators_])
        else:
            self.weights = column_or_1d(weights).reshape(1, len(weights))
            assert (self.weights.shape[1] == self.n_base_estimators_)

            # adjust probability by a factor for integrity （added to 1）
            adjust_factor = self.weights.shape[1] / np.sum(weights)
            self.weights = self.weights * adjust_factor

            print(self.weights)
        return self

    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.base_estimators)

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.base_estimators[index]

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.estimators)

    # noinspection PyMethodParameters
    def _get_param_names(cls):
        # noinspection PyPep8
        """Get parameter names for the estimator
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    # noinspection PyPep8
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        # noinspection PyPep8
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        Returns
        -------
        self : object
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    # @property
    def estimators(self):
        return self.base_estimators, self.fitted_estimators

    # @property
    def base_estimators(self):
        return self.base_estimators

    # @property
    def fitted_estimators(self):
        return self.fitted_estimators
