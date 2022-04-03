from river.linear_model.glm import LinearRegression, LogisticRegression, Perceptron
import ray
from river.compat.river_to_sklearn import convert_river_to_sklearn
__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "Perceptron"
]

# LinearRegression = ray.remote(LinearRegression)

# LinearRegression = convert_river_to_sklearn(LinearRegression)