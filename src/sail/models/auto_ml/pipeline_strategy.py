from sail.models.auto_ml.base_strategy import (
    PipelineActionType,
    PipelineStrategy,
    PipelineActions,
)
from sail.utils.logging import configure_logger

LOGGER = configure_logger()


class DetectAndIncrement(PipelineStrategy):
    def __init__(
        self, search_method, search_data_size, cumulative_scorer, drift_detector, *args
    ) -> None:
        super(DetectAndIncrement, self).__init__(
            search_method, search_data_size, cumulative_scorer, drift_detector
        )

        # Add all pipeline actions
        self.pipeline_actions = PipelineActions()
        self.pipeline_actions.add_action(PipelineActionType.DATA_COLLECTION)
        self.pipeline_actions.add_action(PipelineActionType.FIND_BEST_PIPELINE)
        self.pipeline_actions.add_action(PipelineActionType.SCORE_AND_DETECT_DRIFT)
        self.pipeline_actions.add_action(
            PipelineActionType.PARTIAL_FIT_MODEL,
            next=PipelineActionType.SCORE_AND_DETECT_DRIFT,
        )

        LOGGER.info(
            f"Pipeline Strategy [{self.__class__.__name__}] created with actions: {self.pipeline_actions.get_actions()}"
        )

    def _detect_drift(self):
        value = self.cumulative_scorer.get()
        self.drift_detector.update(value)
        if self.drift_detector.drift_detected:
            LOGGER.info(
                "Drift Detected in the data. Final Estimator will be incrementally trained on the next train()"
            )
            self.pipeline_actions.next()


class DetectAndRetrain(PipelineStrategy):
    def __init__(
        self, search_method, search_data_size, cumulative_scorer, drift_detector, *args
    ) -> None:
        super(DetectAndRetrain, self).__init__(
            search_method, search_data_size, cumulative_scorer, drift_detector
        )

        # Add all pipeline actions
        self.pipeline_actions = PipelineActions()
        self.pipeline_actions.add_action(PipelineActionType.DATA_COLLECTION)
        self.pipeline_actions.add_action(PipelineActionType.FIND_BEST_PIPELINE)
        self.pipeline_actions.add_action(PipelineActionType.SCORE_AND_DETECT_DRIFT)
        self.pipeline_actions.add_action(
            PipelineActionType.FIT_MODEL,
            next=PipelineActionType.SCORE_AND_DETECT_DRIFT,
        )

        LOGGER.info(
            f"Pipeline Strategy [{self.__class__.__name__}] created with actions: {self.pipeline_actions.get_actions()}"
        )

    def _detect_drift(self):
        value = self.cumulative_scorer.get()
        self.drift_detector.update(value)
        if self.drift_detector.drift_detected:
            LOGGER.info(
                "Drift Detected in the data. Final Estimator will be re-trained on the next train()"
            )
            self.pipeline_actions.next()


class DetectAndWarmStart(PipelineStrategy):
    def __init__(
        self,
        search_method,
        search_data_size,
        cumulative_scorer,
        drift_detector,
        *args,
    ) -> None:
        super(DetectAndWarmStart, self).__init__(
            search_method, search_data_size, cumulative_scorer, drift_detector
        )

        # Add all pipeline actions
        self.pipeline_actions = PipelineActions()
        self.pipeline_actions.add_action(PipelineActionType.DATA_COLLECTION)
        self.pipeline_actions.add_action(PipelineActionType.FIND_BEST_PIPELINE)
        self.pipeline_actions.add_action(PipelineActionType.SCORE_AND_DETECT_DRIFT)
        self.pipeline_actions.add_action(PipelineActionType.DATA_COLLECTION)
        self.pipeline_actions.add_action(
            PipelineActionType.PARTIAL_FIT_BEST_PIPELINE,
            next=PipelineActionType.SCORE_AND_DETECT_DRIFT,
        )
        LOGGER.info(
            f"Pipeline Strategy [{self.__class__.__name__}] created with actions: {self.pipeline_actions.get_actions()}"
        )

    def _detect_drift(self):
        value = self.cumulative_scorer.get()
        self.drift_detector.update(value)
        if self.drift_detector.drift_detected:
            LOGGER.info(
                "Drift Detected in the data. Previous SAIL pipelines will be re-evaluated on the next train()"
            )
            self.pipeline_actions.next()


class DetectAndRestart(PipelineStrategy):
    def __init__(
        self, search_method, search_data_size, cumulative_scorer, drift_detector, *args
    ) -> None:
        super(DetectAndRestart, self).__init__(
            search_method, search_data_size, cumulative_scorer, drift_detector
        )

        # Add all pipeline actions
        self.pipeline_actions = PipelineActions()
        self.pipeline_actions.add_action(PipelineActionType.DATA_COLLECTION)
        self.pipeline_actions.add_action(PipelineActionType.FIND_BEST_PIPELINE)
        self.pipeline_actions.add_action(
            PipelineActionType.SCORE_AND_DETECT_DRIFT,
            next=PipelineActionType.DATA_COLLECTION,
        )

        LOGGER.info(
            f"Pipeline Strategy [{self.__class__.__name__}] created with actions: {self.pipeline_actions.get_actions()}"
        )

    def _detect_drift(self):
        value = self.cumulative_scorer.get()
        self.drift_detector.update(value)
        if self.drift_detector.drift_detected:
            LOGGER.info(
                "Drift Detected in the data. SAIL Pipeline wil be re-trained on the next train()"
            )
            self.pipeline_actions.next()


class PeriodicRestart(PipelineStrategy):
    def __init__(
        self, search_method, search_data_size, cumulative_scorer, drift_detector, *args
    ) -> None:
        super(PeriodicRestart, self).__init__(
            search_method, search_data_size, cumulative_scorer, drift_detector
        )

        # Add all pipeline actions
        self.pipeline_actions = PipelineActions()
        self.pipeline_actions.add_action(PipelineActionType.DATA_COLLECTION)
        self.pipeline_actions.add_action(
            PipelineActionType.FIND_BEST_PIPELINE,
            next=PipelineActionType.DATA_COLLECTION,
        )

        LOGGER.info(
            f"Pipeline Strategy [{self.__class__.__name__}] created with actions: {self.pipeline_actions.get_actions()}"
        )
