from __future__ import annotations

from skorch.regressor import NeuralNetRegressor

import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm


class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"

    def __init__(self, output_size: int = 1) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.gap(x))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.0):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(ni, nf, ks, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv1d(nf, nf, ks, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
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


def TemporalConvNet(c_in, layers, ks=2, dropout=0.0):
    temp_layers = []
    for i in range(len(layers)):
        dilation_size = 2 ** i
        ni = c_in if i == 0 else layers[i - 1]
        nf = layers[i]
        temp_layers += [
            TemporalBlock(
                ni,
                nf,
                ks,
                stride=1,
                dilation=dilation_size,
                padding=(ks - 1) * dilation_size,
                dropout=dropout,
            )
        ]
    return nn.Sequential(*temp_layers)


class TCN(nn.Module):
    def __init__(
        self, c_in, c_out, layers=8 * [25], ks=7, conv_dropout=0.0, fc_dropout=0.0
    ):
        super().__init__()
        self.tcn = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
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
        num_inputs (int):
        num_channels (list[int]):
        kernel_size (int):
        dropout (float):
        **kwargs: Arbitrary keyword arguments
    """

    def __init__(self, c_in: int, c_out: int, **kwargs):
        super(TCNRegressor, self).__init__(
            module=TCN,
            module__c_in=c_in,
            module__c_out=c_out,
            train_split=None,
            max_epochs=1,
            batch_size=20,
            **kwargs
        )
