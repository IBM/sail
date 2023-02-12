import typing
from river import neighbors
from river.compat import River2SKLClassifier, River2SKLRegressor

__all__ = ["NearestNeighbors", "KNNClassifier", "KNNRegressor"]


class NearestNeighbors(River2SKLClassifier):
    def __init__(
        self,
        window_size: int,
        min_distance_keep: float,
        distance_func: typing.Union[
            neighbors.base.DistanceFunc, neighbors.base.FunctionWrapper
        ],
    ):
        super(NearestNeighbors, self).__init__(
            river_estimator=neighbors.NearestNeighbors(
                window_size,
                min_distance_keep,
                distance_func,
            )
        )


class KNNClassifier(River2SKLClassifier):
    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        leaf_size: int = 30,
        p: float = 2,
        weighted: bool = True,
        **kwargs
    ):
        super(KNNClassifier, self).__init__(
            river_estimator=neighbors.KNNClassifier(
                n_neighbors, window_size, leaf_size, p, weighted, **kwargs
            )
        )


class KNNRegressor(River2SKLRegressor):
    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        leaf_size: int = 30,
        p: float = 2,
        aggregation_method: str = "mean",
        **kwargs
    ):
        super(KNNRegressor, self).__init__(
            river_estimator=neighbors.KNNRegressor(
                n_neighbors,
                window_size,
                leaf_size,
                p,
                aggregation_method,
                **kwargs
            )
        )
