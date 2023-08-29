from abc import ABC, abstractmethod
from river.compat.river_to_sklearn import SKLEARN_INPUT_X_PARAMS, SKLEARN_INPUT_Y_PARAMS

from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(ABC):
    @abstractmethod
    def fit():
        raise NotImplementedError

    @abstractmethod
    def partial_fit():
        raise NotImplementedError

    @abstractmethod
    def transform():
        raise NotImplementedError

    @abstractmethod
    def fit_transform():
        raise NotImplementedError

    @abstractmethod
    def partial_fit_transform():
        raise NotImplementedError


class SAILTransformer(TransformerMixin, BaseEstimator, BaseTransformer):
    def __init__(self) -> None:
        super(SAILTransformer, self).__init__()
        self.validation_params = {"cast_to_ndarray": True}

    def fit(self, X, y=None):
        # Reset the state if already fitted
        for attr in ("instance_", "n_features_in_"):
            self.__dict__.pop(attr, None)

        # Fit with one pass of the dataset
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        # Check the inputs
        first_call = not hasattr(self, "n_features_in_")
        if y is None:
            X = self._validate_data(
                X,
                **self.validation_params,
                reset=first_call,
                **SKLEARN_INPUT_X_PARAMS,
            )
        else:
            X, y = self._validate_data(
                X,
                y,
                **self.validation_params,
                reset=first_call,
                **SKLEARN_INPUT_X_PARAMS,
                **SKLEARN_INPUT_Y_PARAMS,
            )

        # Fit with one pass of the dataset
        return self._partial_fit(X, y)

    def transform(self, X, y=None, copy=None):
        # Check the inputs
        X = self._validate_data(
            X,
            **self.validation_params,
            reset=False,
            **SKLEARN_INPUT_X_PARAMS,
        )

        # Fit with one pass of the dataset
        return self._transform(X, y, copy)

    def partial_fit_transform(self, X, y=None, **fit_params):
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.partial_fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.partial_fit(X, y, **fit_params).transform(X)
