import river.naive_bayes as naive_bayes
from river.compat import River2SKLClassifier

__all__ = ["GaussianNB", "MultinomialNB", "ComplementNB", "BernoulliNB"]


class GaussianNB(River2SKLClassifier):
    def __init__(self):
        super(GaussianNB, self).__init__(river_estimator=naive_bayes.GaussianNB())


class MultinomialNB(River2SKLClassifier):
    def __init__(self, alpha=1.0):
        super(MultinomialNB, self).__init__(
            river_estimator=naive_bayes.MultinomialNB(alpha)
        )


class ComplementNB(River2SKLClassifier):
    def __init__(self, alpha=1.0):
        super(ComplementNB, self).__init__(
            river_estimator=naive_bayes.ComplementNB(alpha)
        )


class BernoulliNB(River2SKLClassifier):
    def __init__(self, alpha=1.0, true_threshold=0.0):
        super(BernoulliNB, self).__init__(
            river_estimator=naive_bayes.BernoulliNB(alpha, true_threshold)
        )
