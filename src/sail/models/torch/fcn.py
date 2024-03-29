# PyTorch implementation for LSTM FCN for Time Series Classification without LSTM
# Original code in TensorFlow https://github.com/titu1994/LSTM-FCN
# Paper https://arxiv.org/abs/1709.05206
#
# By David Campos and Teodor Vernica

import torch
from torch import nn
from sail.models.torch import SAILTorchClassifier
from sail.models.torch.layers import (
    ConvBlock,
)  # The convolutional block using Conv1D with same padding.


class _FCN(nn.Module):
    def __init__(
        self, in_channels: int, input_size: int, lstm_layers: int = 8, classes: int = 1
    ) -> None:
        """FCN model.
        Args:
            in_channels (int): Number of input channels
            input_size (int): The input size
            lstm_layers (int): Number of hidden LSTM units
            classes (int): Number of classes
        """
        super().__init__()

        self.conv_layers = nn.Sequential(
            *[
                ConvBlock(in_channels, 128, 8, 1),
                ConvBlock(128, 256, 5, 1),
                ConvBlock(256, 128, 3, 1),
            ]
        )

        self.fc = nn.Linear(128, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, 2)

        x_cnn = self.conv_layers(x.transpose(2, 1))
        x_cnn = torch.mean(x_cnn, dim=-1)

        x_out = self.softmax(self.fc(x_cnn))

        return x_out


class FCNClassifier(SAILTorchClassifier):
    def __init__(self, in_channels, input_size, lstm_layers, classes):
        """FCN model.
        Args:
            in_channels (int): Number of input channels
            input_size (int): The input size
            lstm_layers (int): Number of hidden LSTM units
            classes (int): Number of classes
        """
        super(FCNClassifier, self).__init__(
            module=_FCN,
            module__in_channels=in_channels,
            module__input_size=input_size,
            module__lstm_layers=lstm_layers,
            module__classes=classes,
            max_epochs=1,
        )
