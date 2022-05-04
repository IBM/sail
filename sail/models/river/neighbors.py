import river.neighbors.knn_adwin as knn_adwin
import river.neighbors.knn_classifier as knn_classifier
import river.neighbors.knn_regressor as knn_regressor
import river.neighbors.sam_knn as sam_knn
from river.compat import River2SKLClassifier, River2SKLRegressor

__all__ = ["KNNADWINClassifier", "KNNClassifier", "KNNRegressor", "SAMKNNClassifier"]


class KNNADWINClassifier(River2SKLClassifier):
    def __init__(self, n_neighbors=5, window_size=1000, leaf_size=30, p=2):
        super(KNNADWINClassifier, self).__init__(
            river_estimator=knn_adwin.KNNADWINClassifier(
                n_neighbors, window_size, leaf_size, p
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
            river_estimator=knn_classifier.KNNClassifier(
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
            river_estimator=knn_regressor.KNNRegressor(
                n_neighbors, window_size, leaf_size, p, aggregation_method, **kwargs
            )
        )


class SAMKNNClassifier(River2SKLClassifier):
    def __init__(
        self,
        n_neighbors: int = 5,
        distance_weighting=True,
        window_size: int = 5000,
        ltm_size: float = 0.4,
        min_stm_size: int = 50,
        stm_aprox_adaption=True,
        use_ltm=True,
    ):
        super(SAMKNNClassifier, self).__init__(
            river_estimator=sam_knn.SAMKNNClassifier(
                n_neighbors,
                distance_weighting,
                window_size,
                ltm_size,
                min_stm_size,
                stm_aprox_adaption,
                use_ltm,
            )
        )
