# PyTorch implementation for LSTM FCN for Time Series Classification
# Original code in TensorFlow https://github.com/titu1994/LSTM-FCN
# Paper https://arxiv.org/abs/1709.05206
#
# By David Campos and Teodor Vernica

import torch, skorch
from torch import nn
from skorch.classifier import NeuralNetClassifier
from sail.models.torch.layers import ConvBlock # The convolutional block using Conv1D with same padding.


class _LSTM_FCN(nn.Module):

    def __init__(self, in_channels: int, input_size: int, lstm_layers: int = 8, classes: int = 1) -> None:
        """LSTM_FCN model.
        Args:
            in_channels (int): Number of input channels
            input_size (int): The input size
            lstm_layers (int): Number of hidden LSTM units
            classes (int): Number of classes
        """
        super().__init__()
        
        self.lstm = nn.LSTM(in_channels, 128, lstm_layers)
        self.h0 = torch.zeros(lstm_layers, input_size, 128)
        self.c0 = torch.zeros(lstm_layers, input_size, 128)
        self.drop = nn.Dropout(0.8)
        
        self.conv_layers = nn.Sequential(*[
            ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        ])

        self.fc = nn.Linear(256, classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, 2)
        x_rnn, (hn, cn) = self.lstm(x, (self.h0, self.c0))
        x_rnn = self.drop(x_rnn)
        x_rnn = x_rnn[:,-1,:]
        
        x_cnn = self.conv_layers(x.transpose(2,1))
        x_cnn = torch.mean(x_cnn,dim=-1)

        x_all = torch.cat((x_rnn,x_cnn),dim=1)
        x_out = self.softmax(self.fc(x_all))

        return x_out

    
class LSTM_FCN_Classifier(NeuralNetClassifier):
    def __init__(self, in_channels, input_size, lstm_layers, classes):
        """LSTM_FCN model.
        Args:
            in_channels (int): Number of input channels
            input_size (int): The input size
            lstm_layers (int): Number of hidden LSTM units
            classes (int): Number of classes
        """
        super(LSTM_FCN_Classifier, self).__init__(
            module=_LSTM_FCN,
            module__in_channels=in_channels,
            module__input_size=input_size,
            module__lstm_layers=lstm_layers,
            module__classes=classes,
            max_epochs=1)