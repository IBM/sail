from river.compat import River2SKLClassifier, River2SKLRegressor
from sail.utils.mixin import RiverAttributeMixin
from sail.models.base import SAILModel
from sail.utils.logging import configure_logger

LOGGER = configure_logger(logger_name="River")


class RiverBase(SAILModel, RiverAttributeMixin):
    pass


class SailRiverRegressor(River2SKLRegressor, RiverBase):
    def __init__(self, *args, **Kwargs):
        super(SailRiverRegressor, self).__init__(*args, **Kwargs)


class SailRiverClassifier(River2SKLClassifier, RiverBase):
    def __init__(self, *args, **Kwargs):
        super(SailRiverClassifier, self).__init__(*args, **Kwargs)
