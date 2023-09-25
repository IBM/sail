import importlib
import os
from pathlib import Path
from tabnanny import verbose
import torch
import numpy as np
from skorch.classifier import NeuralNetClassifier
from skorch.regressor import NeuralNetRegressor

from sail.models.base import SAILModel
from sail.utils.logging import configure_logger
from sail.utils.serialization import load_obj, save_obj

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


class TorchParamsMixin:
    def fit(self, X, y=None, **fit_params):
        X, y = self.cast_params(X, y)
        return super().fit(X, y, **fit_params)

    def partial_fit(self, X, y=None, classes=None, **fit_params):
        X, y = self.cast_params(X, y)
        return super().partial_fit(X, y, classes, **fit_params)

    def predict(self, X):
        X = self.cast_params(X)
        return super().predict(X)

    def cast_params(self, X, y=None):
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


class SAILTorchRegressor(
    TorchParamsMixin, NeuralNetRegressor, TorchSerializationMixin, SAILModel
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
    TorchParamsMixin, NeuralNetClassifier, TorchSerializationMixin, SAILModel
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
