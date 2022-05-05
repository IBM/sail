import pytest
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sail.utils.stats import nmse
from sail.models.river.linear_model import LinearRegression, LogisticRegression
from sail.models.river.preprocessing import StandardScaler


class TestLinearRegression:
    @pytest.fixture
    def regression_dataset(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        return X, y

    @pytest.fixture
    def classification_dataset(self):
        X, y = datasets.load_breast_cancer(return_X_y=True)
        return X, y

    @pytest.fixture
    def linear_regression(self):
        return LinearRegression()

    @pytest.fixture
    def logistic_regression(self):
        return LogisticRegression()

    def test_linear_regression_nmse(self, regression_dataset, linear_regression):
        X, y = regression_dataset
        X = StandardScaler().fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, train_size=0.7, random_state=99
        )
        model = linear_regression
        for i in range(X_train.shape[0]):
            model.partial_fit(
                X_train[i].reshape(1, -1),
                Y_train[i].reshape(
                    1,
                ),
            )

        Y_test_preds = model.predict(X_test)
        assert Y_test_preds.shape == Y_test.shape

        nmse_val = nmse(Y_test, Y_test_preds)
        print("nmse_val: ", nmse_val)
        assert nmse_val <= 60

    def test_linear_regression_score(self, regression_dataset, linear_regression):
        X, y = regression_dataset
        X = StandardScaler().fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, train_size=0.7, random_state=99
        )
        model = linear_regression
        for i in range(X_train.shape[0]):
            model.partial_fit(
                X_train[i].reshape(1, -1),
                Y_train[i].reshape(
                    1,
                ),
            )

        Y_test_preds = model.predict(X_test)
        r2_score_val = r2_score(Y_test, Y_test_preds)
        print("r2_score_val: ", r2_score_val)
        assert r2_score_val >= 0.38

    def test_logistic_regression_accuracy(
        self, classification_dataset, logistic_regression
    ):
        X, y = classification_dataset
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, train_size=0.7, random_state=99
        )
        model = logistic_regression
        for i in range(X_train.shape[0]):
            model.partial_fit(
                X_train[i].reshape(1, -1),
                Y_train[i].reshape(
                    1,
                ),
                classes=list(set(y)),
            )

        y_pred = model.predict(X_test)
        assert y_pred.shape == Y_test.shape

        accuracy = model.score(X_test, Y_test)
        print("Accuracy: ", accuracy)
        assert accuracy >= 0.85
