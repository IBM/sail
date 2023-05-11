import os

from sail.utils.logging import configure_logger
from sail.utils.serialization import load_obj, save_obj

LOGGER = configure_logger()


class SAILModel:
    def partial_fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

    def save_model(self, model_folder):
        """Saves the sail model to pickle format.

        Args:
            model_folder: String, PathLike, path to SavedModel.
        """
        model_path = os.path.join(model_folder, "model")
        save_obj(
            self,
            model_path,
            str(self.__class__.__name__),
            serialize_type="joblib",
        )
        LOGGER.info("Model saved successfully.")

    def load_model(self, model_folder):
        """Load the sail model from the mentioned folder location

        Args:
            model_folder: One of the following:
                - String or `pathlib.Path` object, path to the saved model

        Returns:
            BaseModel: Model
        """
        model_path = os.path.join(model_folder, "model")
        model = load_obj(
            model_path,
            str(self.__class__.__name__),
            serialize_type="joblib",
        )
        LOGGER.info("Model loaded successfully.")
        return model
