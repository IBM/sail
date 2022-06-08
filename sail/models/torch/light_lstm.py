import torch
import torch.nn as nn
from skorch.regressor import NeuralNetRegressor
from sail.models.torch.base import TorchSerializationMixin


class lightLSTM(nn.Module):
    def __init__(self, ni, nh, nlayers, no):
        super().__init__()
        self.hidden_size = nh
        self.lstm = nn.LSTM(ni, nh, nlayers, batch_first=True)
        self.fc = nn.Linear(nh, no)

    def forward(self, x, hs):
        out, hs = self.lstm(x, hs)  # out.shape = (batch_size, seq_len, hidden_size)
        out = out.view(-1, self.hidden_size)  # out.shape = (seq_len, hidden_size)
        out = self.fc(out)

        return out, hs


class LSTMRegressor(NeuralNetRegressor, TorchSerializationMixin):
    def __init__(self, ni, no, nh, nlayers, module=lightLSTM, **kwargs):
        super(LSTMRegressor, self).__init__(
            module=module,
            module__ni=ni,
            module__no=no,
            module__nh=nh,
            module__nlayers=nlayers,
            **kwargs
        )
