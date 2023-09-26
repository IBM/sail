import os
from pathlib import Path
import tensorflow as tf
from scikeras.wrappers import KerasRegressor, KerasClassifier
import importlib
import inspect
import numpy as np
from scikeras.utils import loss_name
from sail.models.base import SAILModel
from sail.utils.logging import configure_logger
from sail.utils.serialization import load_obj, save_obj

LOGGER = configure_logger(logger_name="TF_Keras")


class KerasSerializationMixin:
    def save_model(self, model_folder):
        """
        Saves Model to Tensorflow SavedModel format.

        Args:
            model_folder: String, PathLike, path to SavedModel.
        """
        if self.model_ is not None:
            self.model_.save(model_folder, overwrite=True, save_format="tf")

            Path(os.path.join(model_folder, ".keras")).touch()

            save_obj(
                obj={
                    "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                },
                location=model_folder,
                file_name="metadata",
                serialize_type="json",
            )

            params = self.get_params()
            params.pop("model")
            params.pop("init_X")
            params.pop("init_y")
            save_obj(
                obj=params,
                location=model_folder,
                file_name="params",
            )

            np.savez(
                os.path.join(model_folder, "init_data"),
                init_X=self.init_X,
                init_y=self.init_y,
            )

            LOGGER.info("Model saved successfully.")

    @classmethod
    def load_model(cls, model_folder):
        """
        Load Model from the mentioned folder location

        Args:
            model_folder: One of the following:
                - String or `pathlib.Path` object, path to the saved model
        """
        if os.path.exists(model_folder):
            keras_model = tf.keras.models.load_model(model_folder)

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
            model_wrapper = model_class(**params)

            model_wrapper.model = keras_model
            init_data = np.load(os.path.join(model_folder, "init_data.npz"))
            model_wrapper.initialize(init_data["init_X"], init_data["init_y"])

            LOGGER.info("Model loaded successfully.")
            return model_wrapper


class KerasInputMixin:
    def _fit(
        self,
        X,
        y,
        sample_weight,
        warm_start: bool,
        epochs: int,
        initial_epoch: int,
        **kwargs,
    ):
        if not self.initialized_:
            self.init_X = X
            self.init_y = y

        if not ((self.warm_start or warm_start) and self.initialized_):
            X, y = self._initialize(X, y)
        else:
            X, y = self._validate_data(X, y)
        self._ensure_compiled_model()

        if sample_weight is not None:
            X, y, sample_weight = self._validate_sample_weight(X, y, sample_weight)

        y = self.target_encoder_.transform(y)
        X = self.feature_encoder_.transform(X)

        self._fit_keras_model(
            X,
            y,
            sample_weight=sample_weight,
            warm_start=warm_start,
            epochs=epochs,
            initial_epoch=initial_epoch,
            **kwargs,
        )


class SAILKerasRegressor(
    KerasInputMixin, KerasRegressor, KerasSerializationMixin, SAILModel
):
    def __init__(self, *args, **kwargs):
        super(SAILKerasRegressor, self).__init__(
            *args,
            **kwargs,
        )


class SAILKerasClassifier(
    KerasInputMixin, KerasClassifier, KerasSerializationMixin, SAILModel
):
    def __init__(self, *args, **kwargs):
        super(SAILKerasClassifier, self).__init__(
            *args,
            **kwargs,
        )
