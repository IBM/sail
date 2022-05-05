from river.compat import River2SKLRegressor, River2SKLClassifier
import river.linear_model as linear_model
import typing
from river import optim


__all__ = [
    "ALMAClassifier",
    "PAClassifier",
    "PARegressor",
    "SoftmaxRegression",
    "LinearRegression",
    "LogisticRegression",
    "Perceptron",
]


class LogisticRegression(River2SKLClassifier):
    def __init__(
        self,
        optimizer: optim.Optimizer = None,
        loss: optim.losses.BinaryLoss = None,
        l2=0.0,
        intercept_init=0.0,
        intercept_lr: typing.Union[float, optim.schedulers.Scheduler] = 0.01,
        clip_gradient=1e12,
        initializer: optim.initializers.Initializer = None,
    ):
        super(LogisticRegression, self).__init__(
            river_estimator=linear_model.LogisticRegression(
                optimizer,
                loss,
                l2,
                intercept_init,
                intercept_lr,
                clip_gradient,
                initializer,
            )
        )


class LinearRegression(River2SKLRegressor):
    def __init__(
        self,
        optimizer: optim.Optimizer = None,
        loss: optim.losses.RegressionLoss = None,
        l2=0.0,
        intercept_init=0.0,
        intercept_lr: typing.Union[optim.schedulers.Scheduler, float] = 0.01,
        clip_gradient=1e12,
        initializer: optim.initializers.Initializer = None,
    ):
        super(LinearRegression, self).__init__(
            river_estimator=linear_model.LinearRegression(
                optimizer,
                loss,
                l2,
                intercept_init,
                intercept_lr,
                clip_gradient,
                initializer,
            )
        )


class Perceptron(River2SKLClassifier):
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


class ALMAClassifier(River2SKLClassifier):
    def __init__(self, p=2, alpha=0.9, B=1 / 0.9, C=2**0.5):
        super(ALMAClassifier, self).__init__(
            river_estimator=linear_model.alma.ALMAClassifier(
                p,
                alpha,
                B,
                C,
            )
        )


class PAClassifier(River2SKLClassifier):
    def __init__(self, C=1.0, mode=1, learn_intercept=True):
        super(PAClassifier, self).__init__(
            river_estimator=linear_model.pa.PAClassifier(C, mode, learn_intercept)
        )


class PARegressor(River2SKLRegressor):
    def __init__(self, C=1.0, mode=1, eps=0.1, learn_intercept=True):
        super(PARegressor, self).__init__(
            river_estimator=linear_model.pa.PARegressor(C, mode, eps, learn_intercept)
        )


class SoftmaxRegression(River2SKLClassifier):
    def __init__(
        self,
        optimizer: optim.Optimizer = None,
        loss: optim.losses.MultiClassLoss = None,
        l2=0,
    ):
        super(SoftmaxRegression, self).__init__(
            river_estimator=linear_model.softmax.SoftmaxRegression(optimizer, loss, l2)
        )
