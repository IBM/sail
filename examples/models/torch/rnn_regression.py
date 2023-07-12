from array import array
from sail.models.torch.rnn import RNNRegressor
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression

n_samples = 6000

stream = make_regression(random_state=1, n_samples=n_samples, n_features=12)

learner_gru = RNNRegressor(
    input_units=12,
    output_units=1,
    hidden_units=20,
    n_hidden_layers=3,
    lr=0.001,
    cell_type="GRU",
)

y_true = array("d")
y_pred = array("d")
index = []
i = 0
wait_samples = 10
# while cnt < n_samples and stream.has_more_samples():
for start in range(0, len(stream[0]), wait_samples):
    end = start + wait_samples

    X, y = stream[0][start:end], stream[1][start:end]
    y = np.array(y)
    y = y.reshape(y.shape[0], -1)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if start != 0:
        y_true.append(y[0])
        y_pred1 = learner_gru.predict(X)[0][0]
        y_pred.append(y_pred1)
        index.append(i)
        i = i + 1

    learner_gru.partial_fit(X, y)
    learner_gru.predict(X)

plt.plot(index, y_pred)
plt.plot(index, y_true)
plt.show()
