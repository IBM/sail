from river.compat import River2SKLClassifier, River2SKLRegressor

from sail.models.base import SAILModel
from sail.utils.logging import configure_logger
from sail.common.mixin import RiverAttributeMixin

LOGGER = configure_logger(logger_name="River")


class SailRiverRegressor(RiverAttributeMixin, River2SKLRegressor, SAILModel):
    def __init__(self, *args, **Kwargs):
        super(SailRiverRegressor, self).__init__(*args, **Kwargs)


class SailRiverClassifier(RiverAttributeMixin, River2SKLClassifier, SAILModel):
    def __init__(self, *args, **Kwargs):
        super(SailRiverClassifier, self).__init__(*args, **Kwargs)
