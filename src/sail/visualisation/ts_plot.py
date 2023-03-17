from typing import Union, List
import matplotlib
import matplotlib.pyplot as plt
import numpy


def plot_series(
    y_orig: Union[List[float], numpy.ndarray],
    y_pred: Union[List[float], numpy.ndarray],
    plot_title="Time Series Plot",
    xaxis_label="Timestep",
    yaxis_label="Value",
    ybound=None,
    save_path=None,
    ts_timestamp=[],
):
    """Plot time series. Plot lines marking each prediction time.

    Parameters
    ----------
    y_orig : Union[List[float], numpy.ndarray]
        Original time series values
    y_pred : Union[List[float], numpy.ndarray]
        Predicted time series values.
    plot_title : str, default=""
        Plot title
    xaxis_label : str, default="Timestep"
        _description_
    yaxis_label : str, default="Value"
        _description_
    ybound : list, default=None
        y-axis range for time series plot.
    save_path : str, default=None
        Location with file name to save the time series plot.
    ts_timestamp : list, default=[]
        Timestamp values to display on the axis.
    """

    total_predictions = len(y_pred)
    font_size = 20
    fig = plt.figure()
    matplotlib.rcParams.update({"font.size": font_size})
    figure_size = (15.4, 7)
    fig.set_size_inches(figure_size)

    plt.title(plot_title)
    plt.ylabel(yaxis_label, fontsize=font_size)
    if len(ts_timestamp) > 0:
        plt.xlabel("Timestamp", fontsize=font_size)
    else:
        plt.xlabel(xaxis_label, fontsize=font_size)

    plt.plot(
        list(range(0, total_predictions)),
        y_pred,
        label="Prediction",
        color="blue",
        linewidth=1.8,
    )
    plt.scatter(
        list(range(0, total_predictions)),
        y_orig,
        label="Truth",
        marker=".",  # type: ignore
        color="green",
        s=60,
    )

    if len(ts_timestamp) > 0:
        timestamp = [i for i in ts_timestamp]
        for i in timestamp:
            plt.axvline(x=i, color="k", ls="dashed", linewidth=2.0)

    plt.legend(loc="upper left", fontsize=25)
    axes = plt.gca()
    axes.set_xlim([1, total_predictions + 10])

    if ybound is None:
        ybound = [0, max(y_orig + y_pred)]
    assert (
        len(ybound) == 2
    ), "ybound must have a lower bound and an upper bound. Ex: [0, 6]"
    axes.set_ylim([ybound[0], ybound[1]])

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    plt.show()
