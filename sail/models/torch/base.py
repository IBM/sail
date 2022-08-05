from sail.utils.logging import configure_logger
import os
from pathlib import Path

LOGGER = configure_logger()


class TorchSerializationMixin:
    def save(self, model_folder):
        """
        Saves the module's parameters, history, and optimizer.

        Args:
            model_folder: String, PathLike, path to model.
        """
        metadata_path = os.path.join(model_folder, "metadata")
        Path(metadata_path).mkdir(parents=True, exist_ok=True)
        self.save_params(  # type: ignore
            f_params=metadata_path + "/model.pkl",
            f_optimizer=metadata_path + "/opt.pkl",
            f_history=metadata_path + "/history.json",
        )
        LOGGER.info("Model saved successfully.")

    def load(self, model_folder):
        """
        Loads the the module's parameters, history, and optimizer.

        Args:
            model_folder: One of the following:
                - String or `pathlib.Path` object, path to the saved model
        """
        metadata_path = os.path.join(model_folder, "metadata")
        if os.path.exists(metadata_path):
            self.load_params(  # type: ignore
                f_params=metadata_path + "/model.pkl",
                f_optimizer=metadata_path + "/opt.pkl",
                f_history=metadata_path + "/history.json",
            )
