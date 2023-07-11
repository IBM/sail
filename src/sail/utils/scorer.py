import importlib
import inspect
import sys
import copy
import river
from river import metrics
from sympy import public

from sail.utils.progress_bar import SAILProgressBar


class SAILModelScorer:
    def __init__(self, scoring=None, estimator_type=None, pipeline_mode=False) -> None:
        self.scoring = scoring
        self.estimator_type = estimator_type
        self.pipeline_mode = pipeline_mode
        self._metric = self._resolve_scoring(scoring, estimator_type)

    @property
    def progressive_score(self):
        return self._metric.get()

    def get_default_scorer(self, estimator_type):
        if estimator_type == "classifier":
            return metrics.Accuracy()
        elif estimator_type == "regressor":
            return metrics.R2()
        elif estimator_type == "clusterer":
            return metrics.Completeness()
        else:
            raise Exception(
                "Invalid Estimator type. Last step in the pipeline can only be a regressor, classifier or clusterer"
            )

    def _resolve_scoring(self, scoring, estimator_type):
        assert not (
            scoring == estimator_type == None
        ), "Either scoring or estimator_type must be a non null value."
        if scoring is None:
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

    @public
    def score(self, y_preds, y_true, sample_weight=1.0, verbose=1):
        desc_type = "Pipeline" if self.pipeline_mode else "Model"
        with SAILProgressBar(
            steps=len(y_preds),
            desc=f"SAIL {desc_type} Score",
            params={
                "Metric": self._metric.__class__.__qualname__,
                "Batch Size": len(y_preds),
            },
            format="scoring",
            verbose=verbose,
        ) as progress:
            scorer = self._resolve_scoring(self.scoring, self.estimator_type)
            for v1, v2 in zip(y_true, y_preds):
                scorer.update(v1, v2, sample_weight)
                progress.update()
            progress.update_params("Score", scorer.get())
        return scorer.get()

    def _eval_progressive_score(
        self, y_preds, y_true, sample_weight=1.0, detached=False, verbose=1
    ):
        desc_type = "Pipeline" if self.pipeline_mode else "Model"
        with SAILProgressBar(
            steps=len(y_preds),
            desc=f"SAIL {desc_type} Progressive Score",
            params={
                "Metric": self._metric.__class__.__qualname__,
                "Batch Size": len(y_preds),
            },
            format="scoring",
            verbose=verbose,
        ) as progress:
            if detached:
                scorer = copy.deepcopy(self._metric)
            else:
                scorer = self._metric
            for v1, v2 in zip(y_true, y_preds):
                scorer.update(v1, v2, sample_weight)
                progress.update()
            progress.update_params("P_Score", scorer.get())
        return scorer.get()

    def clear(self):
        self._metric = self._resolve_scoring(self.scoring, self.estimator_type)
