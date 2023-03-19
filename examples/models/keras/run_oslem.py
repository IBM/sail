import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from sail.models.keras import OSELM

model = OSELM(
    loss="mae",
    optimizer=keras.optimizers.Adam,
    metrics=["accuracy"],
    epochs=1,
    verbose=0,
    num_hidden_nodes=25,
    hidden_layer_activation=tf.nn.sigmoid,
    prediction_window_size=1,
    forgetting_factor=0.5,
)

numLags = 100
predictionStep = 5

df = pd.read_csv(
    "examples/datasets/nyc_taxi.csv",
    header=0,
    names=["time", "data", "timeofday", "dayofweek"],
)
df.head(5)

df.info()
meanSeq = np.mean(df["data"])
stdSeq = np.std(df["data"])
df["data"] = (df["data"] - meanSeq) / stdSeq

df.isnull().sum()

print(df.head())


def getTimeEmbeddedMatrix(sequence, numLags=100, predictionStep=1):
    print("generate time embedded matrix")
    inDim = numLags
    X = np.zeros(shape=(len(sequence), inDim))
    T = np.zeros(shape=(len(sequence), 1))
    for i in range(numLags - 1, len(sequence) - predictionStep):
        X[i, :] = np.array(sequence["data"][(i - numLags + 1) : (i + 1)])
        T[i, :] = sequence["data"][i + predictionStep]
    print("input shape: ", X.shape)
    print("target shape: ", T.shape)
    return (X, T)


(X, T) = getTimeEmbeddedMatrix(df, numLags, predictionStep)

predictions = []
target = []

n_input_nodes = 100
n_hidden_nodes = 25
n_output_nodes = 1


border = int(10 * 1.2 * n_hidden_nodes)
x_train_init = X[:border]
x_train_seq = X[border:]
t_train_init = T[:border]
t_train_seq = T[border:]

X = x_train_seq
T = t_train_seq

for i in range(numLags, 800):
    model.partial_fit(np.array(X[[i], :]), np.array(T[[i], :]), verbose=0)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X[[i], :], T[[i], :])
    ).batch(1)
    Y = model.predict(X[[i + 1], :])
    predictions.append(Y[0][0])
    target.append(T[i][0])
    print(
        "{:5}th timeStep -  target: {:8.4f}   |    prediction: {:8.4f} ".format(
            i, target[-1], predictions[-1]
        )
    )

# Reconstruct original value
predictions = np.array(predictions)
target = np.array(target)
predictions = predictions * stdSeq + meanSeq
target = target * stdSeq + meanSeq


def computeSquareDeviation(predictions, truth):
    squareDeviation = np.square(predictions - truth)
    return squareDeviation


# Calculate NRMSE from skip_eval to the end
skip_eval = 100
squareDeviation = computeSquareDeviation(predictions, target)
squareDeviation[:skip_eval] = None
nrmse = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(predictions)
print("NRMSE {}".format(nrmse))

algorithm = "OSELM"
plt.figure(figsize=(15, 6))
(targetPlot,) = plt.plot(
    target, label="target", color="red", marker=".", linestyle="-"
)
(predictedPlot,) = plt.plot(
    predictions, label="predicted", color="blue", marker=".", linestyle=":"
)
plt.ylabel("value", fontsize=15)
plt.xlabel("time", fontsize=15)
plt.ion()
plt.grid()
plt.legend(handles=[targetPlot, predictedPlot])
plt.title(
    "Time-series Prediction of " + algorithm + " on dataset",
    fontsize=20,
    fontweight=40,
)
plot_path = "./predictionPlot.png"
plt.savefig(plot_path)
plt.draw()
plt.show()
plt.pause(0)
