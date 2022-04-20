"""
Base/Wrapper classes for TensorFlow and Keras based models.
"""
# pyright: reportMissingImports=false

import tensorflow.keras as keras
from scikeras.wrappers import KerasRegressor
from sail.models.base import SAILWrapper


class TFKerasRegressorWrapper(SAILWrapper):
    def save(self, file_path):
        """
        Saves the model to Tensorflow SavedModel or a single HDF5 file.

        Args:
            filepath: String, PathLike, path to SavedModel or H5 file to save the model.
        """
        self.model_.save(file_path)

    def load(self, file_path):
        """
        Loads a model saved via `keras model.save()`.

        Args:
            filepath: One of the following:
                - String or `pathlib.Path` object, path to the saved model
                - `h5py.File` object from which to load the model
        """
        model = keras.models.load_model(file_path)
        return KerasRegressor(model)


class KerasBaseModel(keras.Model):
    def __init__(self) -> None:
        super(KerasBaseModel, self).__init__()

    def compute_loss(self, targets, predictions):
        compiled_loss_fn = self.compiled_loss
        if compiled_loss_fn:
            compiled_loss_fn(targets, predictions)

    def update_metrics(self, targets, predictions):
        compiled_metrics_fn = self.compiled_metrics
        if compiled_metrics_fn:
            compiled_metrics_fn.update_state(targets, predictions)
