from numpy.lib.scimath import sqrt


def nmse(y, yh):
    """MSE normalized by the total count."""

    count = len(y)
    nmse = sqrt(
        sum([(y[i] - yh[i]) * (y[i] - yh[i]) for i in range(count)]) * 1.0 / count
    )
    return nmse
