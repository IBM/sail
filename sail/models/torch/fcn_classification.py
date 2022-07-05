# Implementation of Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline (2016, arXiv) in PyTorch.
# The model is provided by https://github.com/aybchan/time-series-classification 
# The model has been modified slightly changing the stride to 1 by Hao Miao
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier

class _ConvNetModel(nn.Module):
    """ Basic ConvVet: A strong Baseline for Time Series Classification model.

    Args:
        n_in (int): Number of input units
        n_classes (int): Number of classes
    """    
    def __init__(self, n_in, n_classes):
        super(_ConvNetModel,self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(  1,128,(7,1),1,(3,0))
        self.bn1   = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128,256,(5,1),1,(2,0))
        self.bn2   = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256,128,(3,1),1,(1,0))
        self.bn3   = nn.BatchNorm2d(128)

        self.fc4   = nn.Linear(128,self.n_classes)


    def forward(self, x: torch.Tensor):
        x = x.view(-1,1,self.n_in,1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.avg_pool2d(x,1) #modified
        x = torch.mean(x,dim=2)
        x = x.view(-1,128)
        x = self.fc4(x)
        print()
        return F.log_softmax(x,1)
    

class ConvNet(NeuralNetClassifier):
    """Basic TSC model.
    Args:
        n_in (int): Number of input units
        n_classes (int): Number of classes
    """
    def __init__(self, n_in, n_classes):
        super(ConvNet, self).__init__(
            module=_ConvNetModel,
            module__n_in= n_in,
            module__n_classes=n_classes,
            max_epochs=10, 
            lr=0.01, 
            batch_size=12, 
            optimizer=torch.optim.Adam,
            criterion=torch.nn.CrossEntropyLoss,
            # Shuffle training data on each epoch
            iterator_train__shuffle=False )



