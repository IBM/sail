import numpy as np
from pandas import DataFrame


def generate_features_and_targets(time_series: DataFrame, input_column: str):
    """Generate features and targets from a timeseries with a lag of 1.

    Parameters
    ----------
    ts_df: Pandas Dataframe
        time series dataset as a Pandas Dataframe.
    input_column: str
        name of the input column in the dataset.

    Returns
    -------
    ndarray
        Numpy array containing features values.
    ndarray
        Numpy array containing target values. Lag by 1.
    """

    num_rows = len(time_series)
    features = time_series[input_column].astype(float)[0 : num_rows - 1]
    targets = time_series[input_column].astype(float)[1:num_rows]

    return np.array(features).reshape(-1, 1), np.array(targets).reshape(-1, 1)
