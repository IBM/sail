import os
from sail.utils.serialization import save_obj, load_obj
from sail.utils.logging import configure_logger

LOGGER = configure_logger()


def save(model, model_folder):
    """
    Saves the model to joblib format.

    Args:
        model_folder: String, PathLike, path to SavedModel.
    """
    metadata_path = os.path.join(model_folder, "metadata")
    save_obj(model, metadata_path, "model", serialize_type="joblib")
    LOGGER.info("Model saved successfully.")


def load(model_folder):
    """
    Load the Model from the mentioned folder location

    Args:
        model_folder: One of the following:
            - String or `pathlib.Path` object, path to the saved model
    """
    metadata_path = os.path.join(model_folder, "metadata")
    model = load_obj(metadata_path, "model", serialize_type="joblib")
    LOGGER.info("Model loaded successfully.")
    return model
