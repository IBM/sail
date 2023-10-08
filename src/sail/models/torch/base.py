import importlib
import os
from pathlib import Path
import torch
import numpy as np
from skorch.classifier import NeuralNetClassifier
from skorch.regressor import NeuralNetRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils import validation
from sail.models.base import SAILModel
from sail.utils.logging import configure_logger
from sail.utils.serialization import load_obj, save_obj
import copy

LOGGER = configure_logger(logger_name="PyTorch")


class TorchSerializationMixin:
    def save_model(self, model_folder):
        """
        Saves the module's parameters, history, and optimizer.

        Args:
            model_folder: String, PathLike, path to model.
        """
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        self.save_params(  # type: ignore
            f_params=model_folder + "/weights.pkl",
            f_optimizer=model_folder + "/opt.pkl",
            f_history=model_folder + "/history.json",
        )

        save_obj(
            obj={
                "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            },
            location=model_folder,
            file_name="metadata",
            serialize_type="json",
        )

        save_obj(
            obj=self.get_params_for("module"),
            location=model_folder,
            file_name="params",
        )

        Path(os.path.join(model_folder, ".pytorch")).touch()

        LOGGER.info("Model saved successfully.")

    @classmethod
    def load_model(cls, model_folder):
        """
        Loads the the module's parameters, history, and optimizer.

        Args:
            model_folder: One of the following:
                - String or `pathlib.Path` object, path to the saved model
        """
        metadata = load_obj(
            location=model_folder,
            file_name="metadata",
            serialize_type="json",
        )

        params = load_obj(
            location=model_folder,
            file_name="params",
        )

        class_str = metadata["class"].split(".")[-1]
        module_path = ".".join(metadata["class"].split(".")[0:-1])
        model_class = getattr(importlib.import_module(module_path), class_str)
        model = model_class(**params)

        model.initialize()
        if os.path.exists(model_folder):
            model.load_params(  # type: ignore
                f_params=model_folder + "/weights.pkl",
                f_optimizer=model_folder + "/opt.pkl",
                f_history=model_folder + "/history.json",
            )

        return model


class TorchModelMixin:
    def fit(self, X, y=None, **fit_params):
        X, y = self.cast_X_y(X, y)
        return super().fit(X, y, **fit_params)

    def partial_fit(self, X, y=None, classes=None, **fit_params):
        X, y = self.cast_X_y(X, y)
        return super().partial_fit(X, y, classes, **fit_params)

    def predict(self, X):
        X = self.cast_X_y(X)
        return super().predict(X)

    def cast_X_y(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            X = np.asarray(X).astype(np.float32)
        if y is not None:
            if not isinstance(y, torch.Tensor):
                y = np.asarray(y)
                if np.float64 == y.dtype:
                    y = y.astype(np.float32).reshape((-1,))
            return X, y
        else:
            return X

    def check_is_fitted(self, attributes=None, *args, **kwargs):
        """Indicate whether the torch model has been fit."""
        try:
            attributes = (
                attributes or [module + "_" for module in self._modules] or ["module_"]
            )

            validation.check_is_fitted(
                estimator=self, attributes=attributes, *args, **kwargs
            )
            return True
        except NotFittedError:
            return False

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value

        out.pop("module", None)
        for param in copy.deepcopy(out):
            if param.startswith("module__"):
                out.pop(param, None)

        return out


class SAILTorchRegressor(
    TorchModelMixin, NeuralNetRegressor, TorchSerializationMixin, SAILModel
):
    def __init__(self, *args, max_epochs=1, batch_size=-1, train_split=None, **kwargs):
        super(SAILTorchRegressor, self).__init__(
            *args,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_split=train_split,
            **kwargs,
        )


class SAILTorchClassifier(
    TorchModelMixin, NeuralNetClassifier, TorchSerializationMixin, SAILModel
):
    def __init__(
        self,
        *args,
        max_epochs=1,
        batch_size=-1,
        train_split=None,
        **kwargs,
    ):
        super(SAILTorchClassifier, self).__init__(
            *args,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_split=train_split,
            **kwargs,
        )
