import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from torch.nn.parameter import Parameter


class _ONNHBPModel(nn.Module):
    """Online Neural Network with Hedge BackPropagation.
    Online Deep Learning: Learning Deep Neural Networks on the Fly https://arxiv.org/abs/1711.03705

    Paper https://arxiv.org/abs/1711.03705
    Original code in Theano https://github.com/LIBOL/ODL
    Based on Pytorch implementation https://github.com/alison-carrera/onn

    Implemented by: Carlos Muniz Cuza and Jonas Brusokas

        Args:
            input_units: dimensionality of input
            output_units: number of classification classes
            hidden_units: number of hidden units in a single layer
            n_hidden_layers: number of layers in the network
            dropout: dropout coefficient
            beta: beta coefficient (b)
            learning_rate: learning rate for training
            smoothing: smoothing coefficient (s)
    """

    def __init__(self,
                 input_units: int,
                 output_units: int,
                 hidden_units: int,
                 n_hidden_layers: int = 1,
                 dropout: float = 0.2,
                 beta: float = 0.99,
                 learning_rate: float = 0.01,
                 smoothing: float = 0.2,
                 activation: str = 'ReLU',
                 device: str = 'cpu'
                 ):
        super(_ONNHBPModel, self).__init__()

        self.input_units = input_units
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.n_hidden_layers = n_hidden_layers
        # self.batch_size = batch_size

        self.device = torch.device(device)

        self.beta = Parameter(torch.tensor(beta), requires_grad=False).to(self.device)
        self.learning_rate = Parameter(torch.tensor(learning_rate), requires_grad=False).to(self.device)
        self.smoothing = Parameter(torch.tensor(smoothing), requires_grad=False).to(self.device)

        self.loss_array = []

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_units, hidden_units)] +
            [nn.Linear(hidden_units, hidden_units) for _ in range(self.n_hidden_layers - 1)])  #

        self.output_layers = nn.ModuleList([nn.Linear(hidden_units, output_units) for _ in range(
            self.n_hidden_layers)])  #

        self.alpha = Parameter(torch.Tensor(self.n_hidden_layers).fill_(1 / (self.n_hidden_layers + 1)),
                               requires_grad=False)  #

        self.do = nn.Dropout(p=dropout)

        try:
            self.actfn = getattr(nn, activation)()
        except AttributeError:
            print('Invalid activation function: choose ReLu by default')
            self.actfn = nn.ReLU

        self.dtype = torch.float  # ?

    def zero_grad_(self):  #
        for i in range(self.n_hidden_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def __update_alpha(self, losses_per_layer, l: int):  #
        self.alpha[l] *= torch.pow(self.beta, losses_per_layer[l])
        self.alpha[l] = torch.max(self.alpha[l], self.smoothing / self.n_hidden_layers)

    def __update_hidden_layer_weight(self, w, b, l):  #
        self.hidden_layers[l].weight.data -= self.learning_rate * w
        self.hidden_layers[l].bias.data -= self.learning_rate * b

    def calculate_CE_loss(self, X, Y):
        """
        Helper method to calculate the Cross Entropy (CE) loss given ground-truth X and Y
        Args:
            X: ground-truth input
            Y: ground-truth labels/classes

        Returns:
            mean_loss: mean CE loss across all layers
            losses_per_layer: list of CE losses per layer
        """
        predictions_per_layer = self.forward_(X)

        losses_per_layer = []
        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            # loss = criterion(out.view(batch_size, n_classes), Y.view(batch_size).long())
            loss = criterion(out, Y.long())
            losses_per_layer.append(loss)

        mean_loss = torch.stack(losses_per_layer).mean().detach()
        return mean_loss, losses_per_layer

    def update_weights(self, X, Y):
        # batch_size = Y.shape
        # n_classes = self.output_units
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).to(self.device)
        total_predictions_before_update = self.predict_(X)

        mean_loss, losses_per_layer = self.calculate_CE_loss(X, Y)

        w = [0] * self.n_hidden_layers
        b = [0] * self.n_hidden_layers

        with torch.no_grad():
            for i in range(self.n_hidden_layers):
                losses_per_layer[i].backward(retain_graph=True)
                self.output_layers[i].weight.data -= self.learning_rate * self.alpha[i] * self.output_layers[
                    i].weight.grad.data
                self.output_layers[i].bias.data -= self.learning_rate * self.alpha[i] * self.output_layers[
                    i].bias.grad.data

                for j in range(i + 1):
                    w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                    b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad_()

            [self.__update_hidden_layer_weight(w[i], b[i], i) for i in range(self.n_hidden_layers)]
            [self.__update_alpha(losses_per_layer, i) for i in range(self.n_hidden_layers)]

        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        # Return the averaged loss across layers (maybe 'sum' could be more appropriate?)
        return total_predictions_before_update, mean_loss

    def forward(self, X: torch.Tensor):
        scores = self.predict_(X_data=X)
        return scores

    # NOTE: we do not have the 'show_loss' here
    def partial_fit_(self, X_data, Y_data):
        # self.validate_input_X(X_data)
        # self.validate_input_Y(Y_data)
        return self.update_weights(X_data, Y_data)

    def partial_fit(self, X_data, Y_data):
        return self.partial_fit_(X_data, Y_data)

    def forward_(self, X: torch.Tensor):
        """
        Args:
            X: input data
        Returns: [layers, batch, classes] tensor with predictions from each layer
        """
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        X = X.float().to(self.device)
        x = F.relu(self.hidden_layers[0](X))

        hidden_connections = [x]

        for i in range(1, self.n_hidden_layers):
            hidden_connections.append(F.relu(self.hidden_layers[i](hidden_connections[i - 1])))

        output_class = []
        for i in range(self.n_hidden_layers):
            output_class.append(self.output_layers[i](hidden_connections[i]))

        pred_per_layer = torch.stack(output_class)
        return pred_per_layer

    def predict_(self, X_data):
        """
        Args:
            X_data: input data
        Returns: [batch, class_count] tensor with scores for each class
        """
        scores = torch.sum(
            torch.mul(
                self.alpha.view(self.n_hidden_layers, 1).repeat(1, len(X_data))
                    .view(self.n_hidden_layers, len(X_data), 1)
                , self.forward_(X_data))
            , 0)
        return scores.softmax(dim=1)

    def predict(self, X_data):
        """
        Args:
            X_data: input data
        Returns: [batch] tensor with argmax'ed predictions (MAP) for each input item
        """
        scores = self.predict_(X_data)
        return torch.argmax(scores, dim=1)


class ONNHBPClassifier(NeuralNetClassifier):
    def __init__(self,
                 # in_channels, input_size, lstm_layers, classes,
                 input_units: int,
                 output_units: int,
                 hidden_units: int,
                 n_hidden_layers: int = 1,
                 dropout: float = 0.2,
                 beta: float = 0.99,
                 learning_rate: float = 0.01,
                 smoothing: float = 0.2,
                 activation: str = 'ReLU',
                 **kwargs
                 ):
        """
        Online Neural Network trained with Hedge BackPropagation Classifier.

        Paper https://arxiv.org/abs/1711.03705
        Original code in Theano https://github.com/LIBOL/ODL
        Based on Pytorch implementation https://github.com/alison-carrera/onn

        Implemented by: Carlos Muniz Cuza and Jonas Brusokas

        Args:
            input_units: dimensionality of input
            output_units: number of classification classes
            hidden_units: number of hidden units in a single layer
            n_hidden_layers: number of layers in the network
            dropout: dropout coefficient
            beta: beta coefficient (b)
            learning_rate: learning rate for training
            smoothing: smoothing coefficient (s)
            kwargs: NeuralNetClassifier parameters
        """
        super(ONNHBPClassifier, self).__init__(
            module=_ONNHBPModel,
            module__input_units=input_units,
            module__output_units=output_units,
            module__hidden_units=hidden_units,
            module__n_hidden_layers=n_hidden_layers,
            module__dropout=dropout,
            module__beta=beta,
            module__learning_rate=learning_rate,
            module__smoothing=smoothing,
            module__activation=activation,
            max_epochs=1,
            **kwargs
        )

    # OVERRIDES net.py:965
    # This enables training the model EXACTLY as specified in the paper
    # and avoiding all of the defaults imposed by 'skorch'
    def train_step(self, batch, **fit_params):
        step_accumulator = self.get_train_step_accumulator()
        y_pred, loss = self.module_.partial_fit(batch[0], batch[1])
        step = {
            "y_pred": y_pred,
            "loss": loss
        }
        step_accumulator.store_step(step)
        return step_accumulator.get_step()

    # This enables logging "valid_loss"
    def validation_step(self, batch, **fit_params):
        val_loss, _ = self.module_.calculate_CE_loss(batch[0], batch[1])
        y_pred = self.module_.predict_(batch[0])
        return {
            'loss': val_loss,
            'y_pred': y_pred,
        }


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    import numpy as np

    X, Y = make_classification(n_samples=5000, n_features=10, n_informative=4, n_redundant=0, n_classes=10,
                               n_clusters_per_class=1, class_sep=3)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

    print('Training dataset size', X_train.shape)
    print('Testing dataset size', X_test.shape)

    onn_network = ONNHBPClassifier(input_units=10,
                                   output_units=10,
                                   hidden_units=40,
                                   n_hidden_layers=5,
                                   # activation='Tanh',
                                   train_split=None,
                                   verbose=0
                                   )

    partial_fit = onn_network.partial_fit(np.asarray([X_train[0, :]]), np.asarray([y_train[0]]))

    print('Model', partial_fit)

    print("Online Accuracy at the beginning {}".format(partial_fit.score(X_test, y_test)))

    print('Training on more examples ...')
    for i in range(1, 1000):
        partial_fit = onn_network.partial_fit(np.asarray([X_train[i, :]]), np.asarray([y_train[i]]))

    print("Online Accuracy after 1000 examples: {}".format(partial_fit.score(X_test, y_test)))

