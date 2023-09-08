import inspect
import typing


class RiverAttributeMixin:
    def __getattr__(self, __name: str) -> typing.Any:
        return getattr(self.river_estimator, __name)

    def __setattr__(self, __name: str, __value: typing.Any) -> None:
        if __name in list(inspect.signature(__class__).parameters):
            setattr(self.river_estimator, __name, __value)
        else:
            super().__setattr__(__name, __value)
