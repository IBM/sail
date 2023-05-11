import river.naive_bayes as naive_bayes
from sail.models.river.base import SailRiverClassifier


__all__ = ["GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB"]


class GaussianNB(SailRiverClassifier):
    def __init__(self):
        super(GaussianNB, self).__init__(river_estimator=naive_bayes.GaussianNB())


class MultinomialNB(SailRiverClassifier):
    def __init__(self, alpha=1.0):
        super(MultinomialNB, self).__init__(
            river_estimator=naive_bayes.MultinomialNB(alpha)
        )


class ComplementNB(SailRiverClassifier):
    def __init__(self, alpha=1.0):
        super(ComplementNB, self).__init__(
            river_estimator=naive_bayes.ComplementNB(alpha)
        )


class BernoulliNB(SailRiverClassifier):
    def __init__(self, alpha=1.0, true_threshold=0.0):
        super(BernoulliNB, self).__init__(
            river_estimator=naive_bayes.BernoulliNB(alpha, true_threshold)
        )
