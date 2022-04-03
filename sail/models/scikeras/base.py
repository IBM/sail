"""
Base/Wrapper classes for TensorFlow and Keras based models.
"""
# pyright: reportMissingImports=false

import tensorflow.keras as keras
from scikeras.wrappers import KerasRegressor
from sail.models.base import SAILWrapper


class TFKerasRegressorWrapper(KerasRegressor, SAILWrapper):
    """
    TFKerasRegressorWrapper is a base wrapper for all the regression models implemented in Tensorflow 2.5+ / Keras. It inherits KerasRegressor, an implementation of the scikit-learn classifier API for Keras, from the Scikeras package.

    It also inherits SAILWrapper to get access to APIs across the SAIL library.

    Parameters
    ----------

        kerasmodel : Union[None, Callable[..., tf.keras.Model], tf.keras.Model], default=None
            Keras model class. Used to build the Keras Model. When called, must return a compiled instance of a Keras Model to be used by `fit`, `predict`, etc.

        optimizer : Union[str, tf.keras.optimizers.Optimizer, Type[tf.keras.optimizers.Optimizer]], default "sgd"
                This can be a string for Keras' built in optimizersan instance of tf.keras.optimizers.Optimizer or a class inheriting from tf.keras.optimizers.Optimizer. Only strings and classes support parameter routing.

        loss : Union[Union[str, tf.keras.losses.Loss, Type[tf.keras.losses.Loss], Callable], None], default="mse"
            The loss function to use for training. This can be a string for Keras' built in losses, an instance of tf.keras.losses.Loss or a class inheriting from tf.keras.losses.Loss .

        metrics : List[str], default=None
            List of metrics to evaluate and report at each epoch.

        epochs: int, default=1
            Number of training steps.

        verbose: int default=0
            0 means no output printed during training.

    """

    def __init__(
        self,
        model,
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam,
        metrics=None,
        epochs=1,
        verbose=0,
        **kwargs
    ) -> None:
        keras_model = model(**kwargs) if model else None
        super(TFKerasRegressorWrapper, self).__init__(
            keras_model,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            epochs=epochs,
            verbose=verbose,
        )

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
