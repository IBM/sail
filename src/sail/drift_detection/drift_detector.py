import importlib
from typing import Union, Literal
import river
from river.base import DriftDetector, BinaryDriftDetector
from river.drift.binary import EDDM
from river.drift.page_hinkley import PageHinkley
from sail.utils.logging import configure_logger
from sail.common.progress_bar import SAILProgressBar

LOGGER = configure_logger(logger_name="DriftDetector")


class SAILDriftDetector:
    def __init__(
        self,
        model: Union[str, DriftDetector] = EDDM(),
        drift_param: Literal["score", "difference"] = "score",
        verbose=1,
    ) -> None:
        self._drift_detector = self._resolve_drift_detector(model)
        self.drift_param = drift_param
        self.verbose = verbose

    def set_verbose(self, verbose):
        self.verbose = verbose

    def _resolve_drift_detector(self, drift_detector) -> DriftDetector:
        if isinstance(drift_detector, DriftDetector) or isinstance(
            drift_detector, BinaryDriftDetector
        ):
            return drift_detector
        elif isinstance(drift_detector, str):
            if drift_detector == "auto":
                _drift_detector_class = PageHinkley
            elif isinstance(drift_detector, str):
                module = importlib.import_module("river.drift")
                try:
                    _drift_detector_class = getattr(module, drift_detector)
                except AttributeError:
                    raise Exception(
                        f"Drift Detector '{drift_detector}' is not available in River. Available drift detectors: {river.drift.__all__}"
                    )
        else:
            raise TypeError(
                " SAIL Drift Detector `model` must be an instance or str from "
                f"{river.drift.__all__} from river.drift module. Got {drift_detector.__module__}.{drift_detector.__class__.__qualname__}. Set `auto` to use the default."
            )
        return _drift_detector_class()

    def detect_drift(self, score=None, y_pred=None, y_true=None):
        if self.drift_param == "difference":
            return self.detect_drift_with_difference(y_pred, y_true)
        elif self.drift_param == "score":
            return self.detect_drift_with_score(score)

    def detect_drift_with_difference(self, y_preds, y_true):
        with SAILProgressBar(
            steps=len(y_preds),
            desc=f"SAIL Drift detection",
            params={
                "Detector": self._drift_detector.__class__.__qualname__,
                "Batch Size": len(y_preds),
                "Param": "differene",
                "Drift": "No",
            },
            format="scoring",
            verbose=self.verbose,
        ) as progress:
            for yt, yh in zip(y_true, y_preds):
                self._drift_detector.update(yt - yh)
                progress.update()
                if self._drift_detector.drift_detected:
                    progress.update_params("Drift", "Yes")
                    # progress.finish()
                    return True

        return False

    def detect_drift_with_score(self, score):
        with SAILProgressBar(
            steps=1,
            desc=f"SAIL Drift detection",
            params={
                "Detector": self._drift_detector.__class__.__qualname__,
                "Param": "score",
                "Drift": "No",
            },
            format="scoring",
            verbose=self.verbose,
        ) as progress:
            self._drift_detector.update(score)
            progress.update()
            if self._drift_detector.drift_detected:
                progress.update_params("Drift", "Yes")
                return True

        return False
