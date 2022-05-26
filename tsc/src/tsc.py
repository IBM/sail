""" Time series classification
#Implementation of Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline (2016, arXiv) in PyTorch by using a skorch wrapper 
#In this script, we are using two datasets originally used in the paper

#Authors: Marina Georgati, Hao Miao
"""
import torch
from src import model
from skorch import NeuralNetClassifier
import torch

class ConvNet(NeuralNetClassifier):
    """Basic TSC model.
    Args:
        n_in (int): Number of input units
        n_classes (int): Number of classes
    """
    def __init__(self, n_in, n_classes):
        super(ConvNet, self).__init__(
            module=model._ConvNetModel,
            module__n_in= n_in,
            module__n_classes=n_classes,
            max_epochs=10, 
            lr=0.01, 
            batch_size=12, 
            optimizer=torch.optim.Adam,
            criterion=torch.nn.CrossEntropyLoss,
            # Shuffle training data on each epoch
            iterator_train__shuffle=False )



