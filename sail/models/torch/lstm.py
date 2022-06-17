import torch
import torch.nn as nn
from skorch.regressor import NeuralNetRegressor
from sail.models.torch.base import TorchSerializationMixin


class LSTMModel(nn.Module):
    def __init__(self, ni=6, no=3, nh=10, nlayers=1):
        super(LSTMModel, self).__init__()

        self.ni = ni
        self.no = no
        self.nh = nh
        self.nlayers = nlayers

        self.lstms = nn.ModuleList(
            [nn.LSTMCell(self.ni, self.nh)]
            + [nn.LSTMCell(self.nh, self.nh) for i in range(nlayers - 1)]
        )

        self.out = nn.Linear(self.nh, self.no)
        self.do = nn.Dropout(p=0.2)
        self.actfn = nn.Tanh()
        self.device = torch.device("cpu")
        self.dtype = torch.float

    # description of the whole block
    def forward(self, x, h0=None, train=False):
        hs = x  # initiate hidden state
        if h0 is None:
            h = torch.zeros(hs.shape[0], self.nh)
            c = torch.zeros(hs.shape[0], self.nh)
        else:
            (h, c) = h0

        # LSTM cells
        for i in range(self.nlayers):
            h, c = self.lstms[i](hs, (h, c))
            if train:
                hs = self.do(h)
            else:
                hs = h
        y = self.out(hs)
        return y, (h, c)


class ContextlessMSE(torch.nn.MSELoss):
    def forward(self, y_pred, y_true):
        y, (h, c) = y_pred  # extract prediction and context information
        return super().forward(y, y_true)


class LSTMRegressor(NeuralNetRegressor, TorchSerializationMixin):
    def __init__(self, ni, no, nh, nlayers, module=LSTMModel, **kwargs):
        super(LSTMRegressor, self).__init__(
            module=module,
            module__ni=ni,
            module__no=no,
            module__nh=nh,
            module__nlayers=nlayers,
            **kwargs
        )
