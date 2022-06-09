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
        dropout_fn,
        weight_norm: bool,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int,
    ):
        """PyTorch module implementing a residual block module used in `_TCNModule`.

        Parameters
        ----------
        num_filters
            The number of filters in a convolutional layer of the TCN.
        kernel_size
            The size of every kernel in a convolutional layer.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout_fn
            The dropout function to be applied to every convolutional layer.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        nr_blocks_below
            The number of residual blocks before the current one.
        num_layers
            The number of convolutional layers.
        input_size
            The dimensionality of the input time series of the whole network.
        target_size
            The dimensionality of the output time series of the whole network.

        Inputs
        ------
        x of shape `(batch_size, in_dimension, input_chunk_length)`
            Tensor containing the features of the input sequence.
            in_dimension is equal to `input_size` if this is the first residual block,
            in all other cases it is equal to `num_filters`.

        Outputs
        -------
        y of shape `(batch_size, out_dimension, input_chunk_length)`
            Tensor containing the output sequence of the residual block.
            out_dimension is equal to `output_size` if this is the last residual block,
            in all other cases it is equal to `num_filters`.
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


        Parameters
        ----------
        input_size
            The dimensionality of the input time series.
        target_size
            The dimensionality of the output time series.
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        num_layers
            The number of convolutional layers.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout
            The dropout rate for every convolutional layer.
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, input_chunk_length, target_size)`
            Tensor containing the predictions of the next 'output_chunk_length' points in the last
            'output_chunk_length' entries of the tensor. The entries before contain the data points
            leading up to the first prediction, all in chronological order.
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

    def forward(self, x_in):
        x = x_in
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=-1)
        # data is of size (batch_size, input_size, input_chunk_length)
        batch_size = x.size(0)

        for res_block in self.res_blocks_list:
            x = res_block(x)

        x = x.view(batch_size, self.target_size)

        return x


class TCNRegressor(NeuralNetRegressor):
    """Basic TCN model.
    Args:
        c_in (int): number of input channels
        c_out (int): number of output channels
        layers (list): number of channels in each TCN layer
        ks (int): kernel size
        conv_dropout (float): dropout rate for TCN convolutional layers
        fc_dropout (float): dropout rate for FC layers
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
