import torch
import torch.nn as nn
from skorch.regressor import NeuralNetRegressor
from sail.models.torch.base import TorchSerializationMixin


class _RNNModel(nn.Module):
    """Basic RNN/GRU model.

    Args:
        input_units (int): Number of input units
        output_units (int): Number of output units
        hidden_units (int): Number of hidden units
        n_hidden_layers (int): Number of hidden layers
        output_nonlin (torch.nn.Module instance or None (default=nn.Linear)):
            Non-linearity to apply after last layer, if any.
        dropout (float): Dropout
        squeeze_output (bool): default=False
            Whether to squeeze output. For Skorch consistency.
            https://github.com/skorch-dev/skorch/blob/master/skorch/toy.py
        cell_type (string): default="RNN"
    """

    def __init__(
        self,
        input_units,
        output_units,
        hidden_units,
        n_hidden_layers=1,
        dropout=0.2,
        output_nonlin=nn.Linear,
        squeeze_output=False,
        cell_type="RNN",
    ):
        super(_RNNModel, self).__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.output_nonlin = output_nonlin
        self.squeeze_output = squeeze_output
        if cell_type == "RNN":
            rnn = nn.RNNCell
        elif cell_type == "GRU":
            rnn = nn.GRUCell
        else:
            raise ValueError(
                f"RNN type {cell_type} is not supported. supported: [RNN, GRU]"
            )

        self.rnns = nn.ModuleList(
            [rnn(self.input_units, self.hidden_units)]
            + [
                rnn(self.hidden_units, self.hidden_units)
                for i in range(self.n_hidden_layers - 1)
            ]
        )

        if self.output_nonlin:
            self.out = self.output_nonlin(self.hidden_units, self.output_units)
        self.do = nn.Dropout(p=dropout)
        self.actfn = nn.Tanh()
        self.device = torch.device("cpu")
        self.dtype = torch.float

    def forward(self, x, h0=None, train=False):

        hs = x  # initiate hidden state
        if h0 is None:
            h = torch.zeros(hs.shape[0], self.hidden_units)
            c = torch.zeros(hs.shape[0], self.hidden_units)
        else:
            (h, c) = h0

        # RNN cells
        for i in range(self.n_hidden_layers):
            h = self.rnns[i](hs, h)
            if train:
                hs = self.do(h)
            else:
                hs = h
        y = self.out(hs)
        # return y, (h,c)

        return y


class RNNRegressor(NeuralNetRegressor, TorchSerializationMixin):
    """Basic RNN/LSTM/GRU model.

    Args:
        input_units (int): Number of input units
        output_units (int): Number of output units
        hidden_units (int): Number of hidden units
        n_hidden_layers (int): Number of hidden layers
        output_nonlin (torch.nn.Module instance or None (default=nn.Linear)):
            Non-linearity to apply after last layer, if any.
        dropout (float): Dropout
        squeeze_output (bool): default=False
            Whether to squeeze output. Skorch requirement.
        cell_type (string): default="RNN"
        **kwargs: Arbitrary keyword arguments
    """

    def __init__(
        self,
        input_units,
        output_units,
        hidden_units,
        n_hidden_layers=1,
        dropout=0.2,
        output_nonlin=nn.Linear,
        squeeze_output=False,
        cell_type="RNN",
        **kwargs,
    ):
        super(RNNRegressor, self).__init__(
            module=_RNNModel,
            module__input_units=input_units,
            module__output_units=output_units,
            module__hidden_units=hidden_units,
            module__n_hidden_layers=n_hidden_layers,
            module__dropout=dropout,
            module__output_nonlin=output_nonlin,
            module__squeeze_output=squeeze_output,
            module__cell_type=cell_type,
            train_split=None,
            max_epochs=1,
            batch_size=20,
            **kwargs,
        )
