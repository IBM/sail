import importlib
import inspect
import sys

import river
from river import metrics

from sail.utils.progress_bar import SAILProgressBar


class SAILModelScorer:
    def __init__(self, scoring, estimator=None, is_pipeline=False) -> None:
        self.scoring = scoring
        self.estimator = estimator
        self.is_pipeline = is_pipeline
        self._scorer = self._resolve_scoring(scoring, estimator)

    @property
    def progressive_score(self):
        return self._scorer.get()

    def score(self, y_preds, y_true, sample_weight=1.0, verbose=1):
        desc_type = "Pipeline" if self.is_pipeline else "Model"
        with SAILProgressBar(
            steps=len(y_preds),
            desc=f"SAIL {desc_type} Score",
            params={
                "Metric": self._scorer.__class__.__qualname__,
                "Batch Size": len(y_preds),
                "Score": "Calculating...",
            },
            format="scoring",
            verbose=verbose,
        ) as progress:
            scorer = self._resolve_scoring(self.scoring, self.estimator)
            for v1, v2 in zip(y_true, y_preds):
                scorer.update(v1, v2, sample_weight)
                progress.update()
            progress.update_params("Score", scorer.get())
        return scorer.get()

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

    def _resolve_scoring(self, scoring, estimator):
        if scoring is None:
            estimator_type = (
                None if estimator == "passthrough" else estimator._estimator_type
            )
            assert (
                estimator_type is not None
            ), "SAILPipeline.scoring cannot be None when the estimator is set to passthrough in SAILPipeline."
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

    def _eval_progressive_score(self, y_preds, y_true, sample_weight=1.0):
        for v1, v2 in zip(y_true, y_preds):
            self._scorer.update(v1, v2, sample_weight)

        return self._scorer.get()

    def clear(self):
        self._scorer = self._resolve_scoring()
