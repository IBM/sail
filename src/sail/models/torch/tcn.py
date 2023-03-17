# PyTorch implementation for TCN
# This implementation is based on https://github.com/unit8co/darts/blob/master/darts/models/forecasting/tcn_model.py
# Paper https://arxiv.org/abs/1803.01271
# Paper repository: https://github.com/locuslab/TCN
# By Kasper Hjort Berthelsen and Mads Ehrhorn

from skorch.regressor import NeuralNetRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm as wn


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_fn: nn.Dropout,
        weight_norm: bool,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int,
    ):
        """PyTorch module implementing a residual block module used in `TCNModel`.
        This implementation is based on https://github.com/unit8co/darts/blob/master/darts/models/forecasting/tcn_model.py

        Args:
        num_filters (int): The number of filters in a convolutional layer of the TCN.
        kernel_size (int): The size of every kernel in a convolutional layer.
        dilation_base (int): The base of the exponent that will determine the dilation on every level.
        dropout_fn (nn.Dropout): The dropout function to be applied to every convolutional layer.
        weight_norm (bool): Boolean value indicating whether to use weight normalization.
        nr_blocks_below (int): The number of residual blocks before the current one.
        num_layers (int): The number of convolutional layers.
        input_size (int): The dimensionality of the input time series of the whole network.
        target_size (int): The dimensionality of the output time series of the whole network.
        """
        super().__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base ** nr_blocks_below),
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base ** nr_blocks_below),
        )
        if weight_norm:
            self.conv1, self.conv2 = wn(self.conv1), wn(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        """_ResidualBlock forward pass.

        Args:
            x (torch.Tensor): The input data

        Shape:
            - Input: `(batch_size, in_dimension, input_chunk_length)`
            - Output: `(batch_size, out_dimension, input_chunk_length)`
        """
        residual = x

        left_padding = (self.dilation_base ** self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(F.relu(self.conv1(x)))

        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual

        return x


class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        num_filters: int,
        num_layers: int,
        dilation_base: int,
        weight_norm: bool,
        dropout: float,
    ):
        """PyTorch module implementing a dilated TCN module used in `TCNModel`.
        This implementation is based on https://github.com/unit8co/darts/blob/master/darts/models/forecasting/tcn_model.py

        Args:
        input_dim (int): The dimensionality of the input time series.
        output_dim (int): The dimensionality of the output time series.
        kernel_size (int): The size of every kernel in a convolutional layer.
        num_filters (int): The number of filters in a convolutional layer of the TCN.
        num_layers (int): The number of convolutional layers.
        dilation_base (int): The base of the exponent that will determine the dilation on every level.
        weight_norm (bool): Boolean value indicating whether to use weight normalization.
        dropout (float): The dropout rate for every convolutional layer.
        """

        super().__init__()

        # Defining parameters
        self.input_size = input_dim
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_size = output_dim
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(
                num_filters,
                kernel_size,
                dilation_base,
                self.dropout,
                weight_norm,
                i,
                num_layers,
                self.input_size,
                self.target_size,
            )
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    def forward(self, x):
        """TCNModel forward pass.

        Args:
            x (torch.Tensor): The input data

        Shape:
            - Input: `(batch_size, in_dimension, input_chunk_length)`
            - Output: `(batch_size, out_dimension, input_chunk_length)`
        """
        # To aid inputs that omit chunk length, we add a dummy dimension if ndim = 2
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=-1)
        batch_size = x.size(0)

        for res_block in self.res_blocks_list:
            x = res_block(x)

        x = x.view(batch_size, self.target_size)

        return x


class TCNRegressor(NeuralNetRegressor):
    """Basic NeuralNetRegressor wrapper for TCNModel.

    Args:
        input_dim (int): The dimensionality of the input time series.
        output_dim (int): The dimensionality of the output time series.
        kernel_size (int): The size of every kernel in a convolutional layer.
        num_filters (int): The number of filters in a convolutional layer of the TCN.
        num_layers (int): The number of convolutional layers.
        dilation_base (int): The base of the exponent that will determine the dilation on every level.
        weight_norm (bool): Boolean value indicating whether to use weight normalization.
        dropout (float): The dropout rate for every convolutional layer.
        **kwargs: Arbitrary keyword arguments
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        num_filters: int = 3,
        num_layers: int = 3,
        dilation_base: int = 2,
        weight_norm: bool = False,
        dropout: float = 0.2,
        **kwargs
    ):
        super(TCNRegressor, self).__init__(
            module=TCNModel,
            module__input_dim=input_dim,
            module__output_dim=output_dim,
            module__kernel_size=kernel_size,
            module__num_filters=num_filters,
            module__num_layers=num_layers,
            module__dilation_base=dilation_base,
            module__weight_norm=weight_norm,
            module__dropout=dropout,
            train_split=None,
            max_epochs=1,
            batch_size=20,
            **kwargs
        )
