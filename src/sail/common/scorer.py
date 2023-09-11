import copy
import importlib
import inspect
import sys
from typing import Literal
import numpy as np

import river
from river import metrics

from sail.common.progress_bar import SAILProgressBar


class SAILModelScorer:
    def __init__(
        self,
        scoring: str | None = None,
        estimator_type: Literal["regressor", "classifier", "clusterer"] | None = None,
        sample_weight: float = 1.0,
        pipeline_mode: bool = False,
    ) -> None:
        self.scoring = scoring
        self.estimator_type = estimator_type
        self.sample_weight = sample_weight
        self.pipeline_mode = pipeline_mode
        self._metric = self._resolve_scoring(scoring, estimator_type)
        self._scorer_type = "Pipeline" if self.pipeline_mode else "Model"

    @property
    def get_progressive_score(self):
        return self._metric.get()

    def get_state(self):
        if hasattr(self, "_y_true"):
            return {"y_true": self._y_true, "y_pred": self._y_pred}
        else:
            return None

    def set_state(self, state):
        self.progressive_score(state["y_true"], state["y_pred"], verbose=1)

    def get_default_scorer(self, estimator_type):
        if estimator_type == "classifier":
            return metrics.Accuracy()
        elif estimator_type == "regressor":
            return metrics.R2()
        elif estimator_type == "clusterer":
            return metrics.Completeness()
        else:
            raise Exception(
                f"Invalid Estimator type. Estimator can only be a regressor, classifier or clusterer. Given estimator type: {estimator_type}."
            )

    def _resolve_scoring(self, scoring, estimator_type):
        assert not (
            scoring == estimator_type == None
        ), "Either scoring or estimator_type must be a non null value."

        if scoring is None:
            assert (
                estimator_type != "passthrough",
            ), "Scoring cannot be None when the estimator_type is set to passthrough"
            return self.get_default_scorer(estimator_type)

        try:
            if isinstance(scoring, str):
                module = importlib.import_module("river.metrics")
                _scoring_class = getattr(module, scoring)
                return _scoring_class()
            elif isinstance(scoring, metrics.base.Metric):
                return scoring
            elif inspect.isclass(scoring):
                _scoring_class = scoring
                valid_classes = [
                    class_name
                    for _, class_name in inspect.getmembers(
                        sys.modules["river.metrics"], inspect.isclass
                    )
                ]
                if _scoring_class in valid_classes:
                    return _scoring_class()
                else:
                    raise Exception
            else:
                raise Exception

        except:
            method_name = (
                scoring.__name__
                if inspect.isclass(scoring)
                else scoring
                if isinstance(scoring, str)
                else scoring.__class__.__name__
            )
            raise AttributeError(
                f"Method '{method_name}' is not available in river.metrics. Scoring must be a str or an instance of the {river.metrics.__all__}."
            )

    def score(self, y_true, y_pred, sample_weight=None, verbose=1):
        with SAILProgressBar(
            steps=len(y_pred),
            desc=f"SAIL {self._scorer_type} Score",
            params={
                "Metric": self._metric.__class__.__qualname__,
                "Batch Size": len(y_pred),
            },
            format="scoring",
            verbose=verbose,
        ) as progress:
            scorer = self._resolve_scoring(self.scoring, self.estimator_type)
            if not sample_weight:
                sample_weight = self.sample_weight
            for v1, v2 in zip(y_true, y_pred):
                scorer.update(v1, v2, sample_weight)
                progress.update()
            progress.update_params("Score", scorer.get())
        return scorer.get()

    def progressive_score(
        self, y_true, y_pred, sample_weight=None, detached=False, verbose=1
    ):
        with SAILProgressBar(
            steps=len(y_pred),
            desc=f"SAIL {self._scorer_type} Progressive Score",
            params={
                "Metric": self._metric.__class__.__qualname__,
                "Batch Size": len(y_pred),
            },
            format="scoring",
            verbose=verbose,
        ) as progress:
            if not hasattr(self, "_y_true"):
                self._y_true = y_true
                self._y_pred = y_pred
            else:
                self._y_true = np.hstack((self._y_true, y_true))
                self._y_pred = np.hstack((self._y_pred, y_pred))

            if not sample_weight:
                sample_weight = self.sample_weight

            if detached:
                scorer = copy.deepcopy(self._metric)
            else:
                scorer = self._metric

            for v1, v2 in zip(y_true, y_pred):
                scorer.update(v1, v2, sample_weight)
                progress.update()
            progress.update_params("P_Score", scorer.get())
        return scorer.get()

    def clear(self):
        self._metric = self._resolve_scoring(self.scoring, self.estimator_type)
