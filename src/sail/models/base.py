import os
from pathlib import Path

from sklearn.exceptions import NotFittedError
from sklearn.utils import validation

from sail.utils.logging import configure_logger
from sail.utils.serialization import load_obj, save_obj

LOGGER = configure_logger(logger_name="SAILModel")


class SAILModel:
    def fit(self):
        raise NotImplementedError

    def partial_fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

    def check_is_fitted(self, attributes=None, *args, **kwargs):
        """Indicate whether the SAIL model has been fit."""
        try:
            validation.check_is_fitted(
                estimator=self, attributes=attributes, *args, **kwargs
            )
            return True
        except NotFittedError:
            return False

    def save_model(self, model_folder, file_name="model"):
        """Saves the sail model to pickle format.

        Args:
            model_folder: String, PathLike, path to SavedModel.
        """
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        save_obj(
            self,
            model_folder,
            file_name,
        )

        Path(os.path.join(model_folder, ".sailmodel")).touch()

        LOGGER.info("Model saved successfully.")

    @classmethod
    def load_model(cls, model_folder, file_name="model"):
        """Load the sail model from the mentioned folder location

        Args:
            model_folder: One of the following:
                - String or `pathlib.Path` object, path to the saved model

        Returns:
            BaseModel: Model
        """
        model = load_obj(model_folder, file_name)
        LOGGER.info("Model loaded successfully.")

        return model
