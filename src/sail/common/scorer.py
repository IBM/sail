import copy
import importlib
import inspect
import sys
from typing import Literal, Union, Type, List
import numpy as np

import river
from river import metrics

from sail.common.progress_bar import SAILProgressBar


class SAILModelScorer:
    def __init__(
        self,
        scoring: Union[
            List[Union[str, metrics.base.Metric, Type[metrics.base.Metric]]], None
        ] = None,
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
        return self._metric[0].get()

    @property
    def metrics(self):
        metrics = {}
        for metric in self._metric:
            metrics[metric.__class__.__name__] = metric.get()
        return metrics

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
        all_metrics = []
        assert not (
            scoring == estimator_type == None
        ), "Either scoring or estimator_type must be a non null value."

        if scoring is None:
            # fmt: off
            assert  estimator_type != "passthrough", \
                "Scoring cannot be None when the estimator_type is set to passthrough"
            

            all_metrics.append(self.get_default_scorer(estimator_type))
        else:
            assert isinstance(scoring, list), "Scoring must be a list of metric."
            try:
                for score in scoring:
                    if isinstance(score, str):
                        module = importlib.import_module("river.metrics")
                        _scoring_class = getattr(module, score)
                        all_metrics.append(_scoring_class())
                    elif isinstance(score, metrics.base.Metric):
                        all_metrics.append(score)
                    elif inspect.isclass(score):
                        _scoring_class = score
                        valid_classes = [
                            class_name
                            for _, class_name in inspect.getmembers(
                                sys.modules["river.metrics"], inspect.isclass
                            )
                        ]
                        if _scoring_class in valid_classes:
                            all_metrics.append(_scoring_class())
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

        return all_metrics

    def validate_inputs(self, y_true, y_pred):
        return np.asarray(y_true).reshape((-1,)), np.asarray(y_pred).reshape((-1,))

    def score(self, y_true, y_pred, sample_weight=None, verbose=1):
        y_true, y_pred = self.validate_inputs(y_true, y_pred)
        with SAILProgressBar(
            steps=len(y_pred),
            desc=f"SAIL {self._scorer_type} Score",
            params={
                "Metric": self._metric[0].__class__.__qualname__,
                "Batch Size": len(y_pred),
            },
            format="scoring",
            verbose=verbose,
        ) as progress:
            scorer = self._resolve_scoring(self.scoring, self.estimator_type)[0]
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
        y_true, y_pred = self.validate_inputs(y_true, y_pred)

        other_metrics = []
        if len(self._metric) > 1:
            other_metrics = self._metric[1 : len(self._metric)]

        main_score = 0.0
        with SAILProgressBar(
            steps=len(y_pred),
            desc=f"SAIL {self._scorer_type} Progressive Score",
            params={
                "Metric": self._metric[0].__class__.__qualname__,
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
                scorer = copy.deepcopy(self._metric[0])
            else:
                scorer = self._metric[0]

            for v1, v2 in zip(y_true, y_pred):
                scorer.update(v1, v2, sample_weight)
                progress.update()
            progress.update_params("P_Score", scorer.get())
        main_score = scorer.get()

        if not detached:
            for metric in other_metrics:
                if not sample_weight:
                    sample_weight = self.sample_weight

                for v1, v2 in zip(y_true, y_pred):
                    metric.update(v1, v2, sample_weight)

        return main_score

    def clear(self):
        self._metric = self._resolve_scoring(self.scoring, self.estimator_type)
