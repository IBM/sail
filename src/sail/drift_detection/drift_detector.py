import importlib
from typing import Union, Literal
import river
from river.base import DriftDetector
from river.drift import EDDM, PageHinkley

from sail.utils.logging import configure_logger
from sail.utils.progress_bar import SAILProgressBar

LOGGER = configure_logger()


class SAILDriftDetector:
    def __init__(
        self,
        model: Union[str, DriftDetector] = EDDM(),
        drift_param: Literal["score", "difference"] = "score",
    ) -> None:
        self._drift_detector = self._resolve_drift_detector(model)
        self.drift_param = drift_param

    def _resolve_drift_detector(self, drift_detector) -> DriftDetector:
        if isinstance(drift_detector, DriftDetector):
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
                f"{river.drift.__all__} from river.drift module. Got {drift_detector.__module__}.{drift_detector.__qualname__}. Set `auto` to use the default."
            )
        return _drift_detector_class()

    def detect_drift(self, *args):
        if self.drift_param == "difference":
            return self._detect_drift_with_difference(args[1], args[2])
        elif self.drift_param == "score":
            return self._detect_drift_with_score(args[0])

    def _detect_drift_with_difference(self, y_preds, y_true):
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
            verbose=1,
        ) as progress:
            for yt, yh in zip(y_true, y_preds):
                self._drift_detector.update(yt - yh)
                progress.update()
                if self._drift_detector.drift_detected:
                    progress.update_params("Drift", "Yes")
                    # progress.finish()
                    return True

        return False

    def _detect_drift_with_score(self, score):
        with SAILProgressBar(
            steps=1,
            desc=f"SAIL Drift detection",
            params={
                "Detector": self._drift_detector.__class__.__qualname__,
                "Param": "score",
                "Drift": "No",
            },
            format="scoring",
            verbose=1,
        ) as progress:
            self._drift_detector.update(score)
            progress.update()
            if self._drift_detector.drift_detected:
                progress.update_params("Drift", "Yes")
                return True

        return False
