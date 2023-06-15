import river.linear_model as linear_model
import typing
from river import optim
from sail.models.river.base import SailRiverClassifier, SailRiverRegressor

__all__ = [
    "ALMAClassifier",
    "PAClassifier",
    "PARegressor",
    "SoftmaxRegression",
    "LinearRegression",
    "LogisticRegression",
    "Perceptron",
]


class LogisticRegression(SailRiverClassifier):
    def __init__(
        self,
        optimizer: optim.base.Optimizer = None,
        loss: optim.losses.BinaryLoss = None,
        l2=0.0,
        l1=0.0,
        intercept_init=0.0,
        intercept_lr: typing.Union[float, optim.base.Scheduler] = 0.01,
        clip_gradient=1e12,
        initializer: optim.base.Initializer = None,
    ):
        super(LogisticRegression, self).__init__(
            river_estimator=linear_model.LogisticRegression(
                optimizer,
                loss,
                l2,
                l1,
                intercept_init,
                intercept_lr,
                clip_gradient,
                initializer,
            )
        )


class LinearRegression(SailRiverRegressor):
    def __init__(
        self,
        optimizer: optim.base.Optimizer = None,
        loss: optim.losses.RegressionLoss = None,
        l2=0.0,
        l1=0.0,
        intercept_init=0.0,
        intercept_lr: typing.Union[optim.base.Scheduler, float] = 0.01,
        clip_gradient=1e12,
        initializer: optim.base.Initializer = None,
    ):
        super(LinearRegression, self).__init__(
            river_estimator=linear_model.LinearRegression(
                optimizer,
                loss,
                l2,
                l1,
                intercept_init,
                intercept_lr,
                clip_gradient,
                initializer,
            )
        )


class Perceptron(SailRiverClassifier):
    def __init__(
        self,
        l2=0.0,
        clip_gradient=1e12,
        initializer: optim.initializers.Initializer = None,
    ):
        super(Perceptron, self).__init__(
            river_estimator=linear_model.Perceptron(
                l2,
                clip_gradient,
                initializer,
            )
        )


class ALMAClassifier(SailRiverClassifier):
    def __init__(self, p=2, alpha=0.9, B=1 / 0.9, C=2**0.5):
        super(ALMAClassifier, self).__init__(
            river_estimator=linear_model.alma.ALMAClassifier(
                p,
                alpha,
                B,
                C,
            )
        )


class PAClassifier(SailRiverClassifier):
    def __init__(self, C=1.0, mode=1, learn_intercept=True):
        super(PAClassifier, self).__init__(
            river_estimator=linear_model.pa.PAClassifier(C, mode, learn_intercept)
        )


class PARegressor(SailRiverRegressor):
    def __init__(self, C=1.0, mode=1, eps=0.1, learn_intercept=True):
        super(PARegressor, self).__init__(
            river_estimator=linear_model.pa.PARegressor(C, mode, eps, learn_intercept)
        )


class SoftmaxRegression(SailRiverClassifier):
    def __init__(
        self,
        optimizer: optim.base.Optimizer = None,
        loss: optim.losses.MultiClassLoss = None,
        l2=0,
    ):
        super(SoftmaxRegression, self).__init__(
            river_estimator=linear_model.softmax.SoftmaxRegression(optimizer, loss, l2)
        )
