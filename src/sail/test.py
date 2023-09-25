import numpy as np
import torch
from sail.models.torch.os_cnn import OS_CNN_CLassifier
from sklearn.datasets import make_classification


def load_to_torch(X_train, y_train, device):
    X_train = torch.from_numpy(X_train)
    X_train.requires_grad = False
    X_train = X_train.to(device)
    y_train = torch.from_numpy(y_train).reshape((-1,)).to(device)

    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze_(1)
    return X_train, y_train


X, y = make_classification(100, 270, random_state=0)
X, y = X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)

# the model prints out the result every epoch
X_train, y_train = X, y
X_train, y_train = load_to_torch(X_train, y_train, "cpu")
Max_kernel_size = 89
start_kernel_size = 1
input_channel = X_train.shape[1]  # input channel size
n_class = max(y_train) + 1  # output class number
receptive_field_shape = min(int(X_train.shape[-1] / 4), Max_kernel_size)
net = OS_CNN_CLassifier(2, input_channel, receptive_field_shape)

for i in range(0, 3):
    net.partial_fit(X_train, y_train)
