from __future__ import annotations

from skorch.regressor import NeuralNetRegressor

import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm


class _GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"

    def __init__(self, output_size: int = 1) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.gap(x))


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super(_Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.0):
        """Initialises the Temporal Block

        Args:
            ni (int): number of inputs
            nf (int): number of outputs
            ks (int): kernel size
            stride (int): stride lengths
            dilation (int): dilation
            padding (int or str): padding
            dropout (float, optional): dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(ni, nf, ks, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv1d(nf, nf, ks, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(ni, nf, 1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(_TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                _TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class _TCN(nn.Module):
    def __init__(self, c_in, c_out, layers, ks, conv_dropout, fc_dropout):
        super().__init__()
        self.tcn = _TemporalConvNet(c_in, layers, kernel_size=ks, dropout=conv_dropout)
        self.gap = _GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1], c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)


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
        c_in: int,
        c_out: int,
        layers=8 * [25],
        ks=7,
        conv_dropout=0.1,
        fc_dropout=0.1,
        **kwargs
    ):
        super(TCNRegressor, self).__init__(
            module=_TCN,
            module__c_in=c_in,
            module__c_out=c_out,
            module__layers=layers,
            module__ks=ks,
            module__conv_dropout=conv_dropout,
            module__fc_dropout=fc_dropout,
            train_split=None,
            max_epochs=1,
            batch_size=20,
            **kwargs
        )
