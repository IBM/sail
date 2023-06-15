import copy
import inspect
import typing

from river.compat import (
    River2SKLClassifier,
    River2SKLRegressor,
    river_to_sklearn,
)
from sklearn import preprocessing, utils

from sail.models.base import SAILModel
from sail.utils.logging import configure_logger

LOGGER = configure_logger()


class RiverMixin:
    def __getattr__(self, __name: str) -> typing.Any:
        return getattr(self.river_estimator, __name)

    def __setattr__(self, __name: str, __value: typing.Any) -> None:
        if __name in list(inspect.signature(__class__).parameters):
            setattr(self.river_estimator, __name, __value)
        else:
            super().__setattr__(__name, __value)


class RiverBase(SAILModel, RiverMixin):
    pass


class SailRiverRegressor(River2SKLRegressor, RiverBase):
    def __init__(self, *args, **Kwargs):
        super(SailRiverRegressor, self).__init__(*args, **Kwargs)


class SailRiverClassifier(River2SKLClassifier, RiverBase):
    def __init__(self, *args, **Kwargs):
        super(SailRiverClassifier, self).__init__(*args, **Kwargs)
