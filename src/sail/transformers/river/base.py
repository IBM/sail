import copy
from river.compat.river_to_sklearn import SKLEARN_INPUT_X_PARAMS, SKLEARN_INPUT_Y_PARAMS
import numpy as np
from river import base
from river.compat import River2SKLTransformer, river_to_sklearn
from sklearn import utils
from sail.utils.mixin import RiverAttributeMixin
from sail.transformers.base import BaseTransformer


class BaseRiverTransformer(RiverAttributeMixin, River2SKLTransformer):
    def __init__(self, *args, **Kwargs):
        super(BaseRiverTransformer, self).__init__(*args, **Kwargs)
        self.validation_params = {"cast_to_ndarray": True}

    def _partial_fit(self, X, y=None):
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

        # scikit-learn's convention is that fit shouldn't mutate the input parameters; we have to deep copy the provided estimator in order to respect this convention
        if not hasattr(self, "instance_"):
            self.instance_ = copy.deepcopy(self.river_estimator)

        # Call learn_one for each observation
        if isinstance(self.instance_, base.SupervisedTransformer):
            for x, yi in river_to_sklearn.STREAM_METHODS[type(X)](X, y):
                self.instance_.learn_one(x, yi)
        else:
            for x, _ in river_to_sklearn.STREAM_METHODS[type(X)](X):
                self.instance_.learn_one(x)

        return self

    def transform(self, X):
        """Predicts the target of an entire dataset contained in memory.

        Parameters
        ----------
        X
            array-like of shape (n_samples, n_features)

        Returns
        -------
        Transformed output.

        """

        # Check the fit method has been called
        utils.validation.check_is_fitted(self, attributes="instance_")

        # Check the input
        X = self._validate_data(
            X, **self.validation_params, reset=False, **SKLEARN_INPUT_X_PARAMS
        )

        # Call predict_proba_one for each observation
        X_trans = [None] * len(X)
        for i, (x, _) in enumerate(river_to_sklearn.STREAM_METHODS[type(X)](X)):
            X_trans[i] = list(self.instance_.transform_one(x).values())

        return np.asarray(X_trans)

    def partial_fit_transform(self, X, y=None, **fit_params):
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.partial_fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.partial_fit(X, y, **fit_params).transform(X)
