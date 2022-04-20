from array import array
from skmultiflow.data import RegressionGenerator
from sail.models.torch.rnn import RNNRegressor
import matplotlib.pyplot as plt
import numpy as np

n_samples = 6000

stream = RegressionGenerator(random_state=1,
                             n_samples=n_samples,
                             n_features=12)
learner_gru = RNNRegressor(input_units=12, output_units=1, hidden_units=20,
                           n_hidden_layers=3, lr=0.001, cell_type="GRU")

cnt = 0
y_true = array('d')
y_pred = array('d')
index = []
i = 0
wait_samples = 10
while cnt < n_samples and stream.has_more_samples():
    X, y = stream.next_sample(batch_size=wait_samples)
    y = np.array(y)
    y = y.reshape(y.shape[0], -1)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Test every n samples
    if (cnt % wait_samples == 0) & (cnt != 0):
        y_true.append(y[0])
        y_pred1 = learner_gru.predict(X)[0][0]
        y_pred.append(y_pred1)
        index.append(i)
        i = i + 1
    learner_gru.partial_fit(X, y)
    learner_gru.predict(X)
    cnt += 1

plt.plot(index, y_pred)
plt.plot(index, y_true)
plt.show()
