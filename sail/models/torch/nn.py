import torch
import torch.nn as nn
from skorch.regressor import NeuralNetRegressor
import torch.nn.functional as F
from sail.models.torch.base import TorchSerializationMixin


class NNModel(nn.Module):
    def __init__(
        self,
        num_units=10,
        nonlin=F.relu,
    ):
        super(NNModel, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X


class NNRegressor(NeuralNetRegressor, TorchSerializationMixin):
    def __init__(self, ni, no, nh, nlayers, module=NNModel, **kwargs):
        super(NNRegressor, self).__init__(
            module=module,
            module__ni=ni,
            module__no=no,
            module__nh=nh,
            module__nlayers=nlayers,
            **kwargs
        )


if __name__ == "__main__":
    regressor = NNRegressor(None, None, None, None)
