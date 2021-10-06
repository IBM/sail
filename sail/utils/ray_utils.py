import ray
import numpy as np
import numpy.typing as npt

"""
Ray functions for distributed training and scoring
"""

@ray.remote
def _model_fit(model: object, X: npt.NDArray, y: npt.ArrayLike, classes: list = None):
    """
    :param model: Any machine learning model with partial_fit function defined.
    :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
    :param y: numpy.ndarray of shape (n_samples)
           Labels for the target variable.
    :param classes: numpy.ndarray, optional.
           Unique classes in the data y.
    :return: Trained machine learning model.
    """
    if classes is None:
        trained_model = model.partial_fit(X, y)
    else:
        trained_model = model.partial_fit(X, y, classes)
    return trained_model


@ray.remote
def _model_fit_classifier(model: object, X: npt.NDArray, y: npt.ArrayLike, classes: list = None):
    if classes is None:
        classes = np.unique(y)
    trained_model = model.partial_fit(X, y, classes)
    return trained_model


@ray.remote
def _model_predict(model: object, X: npt.NDArray):
    """
    :param model: Any machine learning model with partial_fit function defined.
    :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
    :return: numpy.ndarray of shape (n_samples)
        Predictions for the samples in X.
    """
    return model.predict(X)


@ray.remote
def _model_metrics(model: object, X: npt.NDArray, y_true: npt.ArrayLike, metric: object):
    """
    :param model: Any machine learning model with partial_fit function defined.
    :param X: numpy.ndarray of shape (n_samples, n_features).
               Input samples.
    :param y_true: numpy.ndarray of shape (n_samples)
           True labels or ground truth for the samples in X
    :param metric: Object that defines the quality of predictions
           (ex. metrics.accuracy_score in scikit-learn)
    :return: Metric as defined by the metric object.
    """
    y_pred = model.predict(X)
    return metric(y_true, y_pred)
