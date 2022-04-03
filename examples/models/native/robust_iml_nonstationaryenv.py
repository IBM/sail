import numpy as np
import matplotlib.pyplot as plt
from sail.models.native.robust_iml import RoIML_Model
from sail.models.native.robust_iml import sigmoidActivation

# load artificial training dataset 1
np.random.seed(0)
input_variables = np.random.normal(0, 0.1, 4)

multiplier = np.zeros((500, 4))
multiplier[0] = [0.5, 0.2, 0.7, 0.8]

for i in range(1, 500):
    for j in range(4):
        multiplier[i][j] = multiplier[i - 1][j] + (
            (i * sigmoidActivation(multiplier[i - 1][j])) / 10**4.7
        )

X = multiplier * input_variables
d = np.sum(X, axis=1)
d = d.reshape((d.shape[0], 1))

# adding noise to output y
noise = np.random.normal(0, 0.05, d.shape)
d = d + noise

num_samples = X.shape[0]
num_x_features = X.shape[1]
num_output = d.shape[1]

# load artificial test dataset 1

np.random.seed(0)
input_variables = np.random.normal(0, 0.1, 4)

multiplier = np.zeros((500, 4))
multiplier[0] = [0.5, 0.2, 0.7, 0.8]

for i in range(1, 500):
    for j in range(4):
        multiplier[i][j] = multiplier[i - 1][j] + (
            (i * sigmoidActivation(multiplier[i - 1][j])) / 10**4.7
        )


X_test = multiplier * input_variables
d_test = np.sum(X_test, axis=1)
d_test = d.reshape((d.shape[0], 1))
noise = np.random.normal(0, 0.05, d.shape)
d_test = d_test + noise

model = RoIML_Model(num_x_features, num_output)

mse = []
pred_value = []
des_output = []

for s in range(num_samples):
    for j in range(num_output):
        model.partial_fit(s, X, d[s][j], j)
        y_pred = model.predict(X_test[s])
        pred_value.append(y_pred)
        des_output.append(d_test[s])
        mse.append((np.square(np.array(pred_value) - np.array(des_output)).mean()))


# plotting the mean square error on test set
x_axis = range(num_samples)
plt.plot(x_axis, mse)
plt.xlabel("Number of Samples")
plt.ylabel("MSE")
plt.show()
