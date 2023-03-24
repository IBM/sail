import os
import tensorflow.keras as keras
from sail.utils.serialization import save_obj, load_obj
from sail.utils.logging import configure_logger
from scikeras.wrappers import KerasRegressor

LOGGER = configure_logger()


class KerasSerializationMixin:
    def save(self, model_folder):
        """
        Saves Model to Tensorflow SavedModel format.

        Args:
            model_folder: String, PathLike, path to SavedModel.
        """
        if self.model_ is not None:
            metadata_path = os.path.join(model_folder, "metadata")
            self.model_.save(metadata_path)
            # self.save_metrics(model_folder)
            LOGGER.info("Model saved successfully.")

    def load(self, model_folder):
        """
        Load Model from the mentioned folder location

        Args:
            model_folder: One of the following:
                - String or `pathlib.Path` object, path to the saved model
        """
        metadata_path = os.path.join(model_folder, "metadata")
        if os.path.exists(metadata_path):
            self.model = keras.models.load_model(metadata_path)
            # self.model_.metrics = self.load_metrics(model_folder)
            LOGGER.info("Model loaded successfully.")

    # def save_metrics(self, model_folder: str):
    #     try:
    #         metrics_path = os.path.join(model_folder, "metrics")
    #         for index, metric in enumerate(self.model_.metrics):
    #             name = metric.name + "_" + str(index)
    #             save_obj(
    #                 obj={"class": metric.__class__, "config": metric.get_config()},
    #                 location=metrics_path,
    #                 file_name=name,
    #             )
    #             LOGGER.info("Metric %s saved successfully.." % (name))
    #     except Exception as e:
    #         LOGGER.info("Not able to save metrics")

    # def load_metrics(self, model_folder: str):
    #     metrics = []
    #     try:
    #         metrics_path = os.path.join(model_folder, "metrics")
    #         for filename in os.listdir(metrics_path):
    #             filename = filename.split(".")[0]
    #             metric_obj = load_obj(location=metrics_path, file_name=filename)
    #             metric = metric_obj["class"].from_config(metric_obj["config"])
    #             metrics.append(metric)
    #             LOGGER.info("Metric %s loaded successfully.." % (metric.name))
    #     except Exception as e:
    #         LOGGER.info("Not able to load metrics")
    #     return metrics
