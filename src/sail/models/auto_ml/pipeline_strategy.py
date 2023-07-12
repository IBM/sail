from sail.models.auto_ml.base_strategy import (
    PipelineActionType,
    PipelineStrategy,
    PipelineActions,
)
from sail.utils.logging import configure_logger

LOGGER = configure_logger()


class DetectAndIncrement(PipelineStrategy):
    def __init__(
        self,
        search_method,
        search_data_size,
        drift_detector,
        **kwargs,
    ) -> None:
        super(DetectAndIncrement, self).__init__(
            search_method, search_data_size, drift_detector, **kwargs
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

    def _detect_drift(self, *args):
        if self.drift_detector.detect_drift(*args):
            LOGGER.info(
                "Drift Detected in the data. Final Estimator will be incrementally trained on the next train()"
            )
            self.pipeline_actions.next()
            return True
        return False


class DetectAndRetrain(PipelineStrategy):
    def __init__(
        self,
        search_method,
        search_data_size,
        drift_detector,
        **kwargs,
    ) -> None:
        super(DetectAndRetrain, self).__init__(
            search_method, search_data_size, drift_detector, **kwargs
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

    def _detect_drift(self, *args):
        if self.drift_detector.detect_drift(*args):
            LOGGER.info(
                "Drift Detected in the data. Final Estimator will be re-trained on the next train()"
            )
            self.pipeline_actions.next()
            return True
        return False


class DetectAndWarmStart(PipelineStrategy):
    def __init__(
        self,
        search_method,
        search_data_size,
        drift_detector,
        **kwargs,
    ) -> None:
        super(DetectAndWarmStart, self).__init__(
            search_method, search_data_size, drift_detector, **kwargs
        )

        # Add all pipeline actions
        self.pipeline_actions = PipelineActions()
        self.pipeline_actions.add_action(PipelineActionType.DATA_COLLECTION)
        self.pipeline_actions.add_action(PipelineActionType.FIND_BEST_PIPELINE)
        self.pipeline_actions.add_action(PipelineActionType.SCORE_AND_DETECT_DRIFT)
        self.pipeline_actions.add_action(PipelineActionType.DATA_COLLECTION)
        self.pipeline_actions.add_action(
            PipelineActionType.WARM_START_FIND_BEST_PIPELINE,
            next=PipelineActionType.SCORE_AND_DETECT_DRIFT,
        )
        LOGGER.info(
            f"Pipeline Strategy [{self.__class__.__name__}] created with actions: {self.pipeline_actions.get_actions()}"
        )

    def _detect_drift(self, *args):
        if self.drift_detector.detect_drift(*args):
            LOGGER.info(
                "Drift Detected in the data. SAIL AutoML will re-start with previously evaluated configurations on the next train()"
            )
            self.pipeline_actions.next()
            return True
        return False


class DetectAndRestart(PipelineStrategy):
    def __init__(
        self,
        search_method,
        search_data_size,
        drift_detector,
        **kwargs,
    ) -> None:
        super(DetectAndRestart, self).__init__(
            search_method, search_data_size, drift_detector, **kwargs
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

    def _detect_drift(self, *args):
        if self.drift_detector.detect_drift(*args):
            LOGGER.info(
                "Drift Detected in the data. SAIL AutoML will re-start from scratch on the next train()"
            )
            self.pipeline_actions.next()
            return True
        return False


class PeriodicRestart(PipelineStrategy):
    def __init__(
        self,
        search_method,
        search_data_size,
        drift_detector,
        **kwargs,
    ) -> None:
        super(PeriodicRestart, self).__init__(
            search_method, search_data_size, drift_detector, **kwargs
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


class PrequentialTraining(PipelineStrategy):
    def __init__(
        self,
        search_method,
        search_data_size,
        **kwargs,
    ) -> None:
        kwargs.pop("incremental_training")
        super(PrequentialTraining, self).__init__(
            search_method,
            search_data_size,
            incremental_training=True,
            **kwargs,
        )

        # Add all pipeline actions
        self.pipeline_actions = PipelineActions()
        self.pipeline_actions.add_action(PipelineActionType.DATA_COLLECTION)
        self.pipeline_actions.add_action(PipelineActionType.FIND_BEST_PIPELINE)
        self.pipeline_actions.add_action(
            PipelineActionType.PARTIAL_FIT_PIPELINE,
            next=PipelineActionType.PARTIAL_FIT_PIPELINE,
        )

        LOGGER.info(
            f"Pipeline Strategy [{self.__class__.__name__}] created with actions: {self.pipeline_actions.get_actions()}"
        )
