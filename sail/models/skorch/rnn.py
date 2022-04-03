import torch
import torch.nn as nn
from skorch.regressor import NeuralNetRegressor


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates an RNN instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out


class ContextlessMSE(torch.nn.MSELoss):
    def forward(self, y_pred, y_true):
        y, (h, c) = y_pred # extract prediction and context information
        return super().forward(y, y_true)



class RNNRegressor:
    def __init__(self, input_dim, output_dim, hidden_dim, layer_dim):
        self.net = NeuralNetRegressor(
    module=RNNModel,
    module__input_dim=input_dim,
    module__output_dim=output_dim,
    module__hidden_dim=hidden_dim,
    module__layer_dim=layer_dim,
    max_epochs=20,
    lr=0.1
    #     device='cuda',  # uncomment this to train with CUDA
    )

    def fit(self, X, y):
        self.net.fit(X, y)