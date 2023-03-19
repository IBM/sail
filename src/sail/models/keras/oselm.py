"""
Keras wrapper for The Online Sequential Extreme Learning Machine (OSELM).
"""

import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from tensorflow.python.keras.engine import data_adapter

from sail.models.keras.base import KerasSerializationMixin


class _Model(tf.keras.Model):
    def __init__(
        self,
        num_hidden_nodes: int = 100,
        hidden_layer_activation: str = "sigmoid",
        prediction_window_size: int = 1,
        forgetting_factor: float = 0.9,
    ):
        super(_Model, self).__init__()
        self.prediction_window_size = prediction_window_size
        self.forgetting_factor = forgetting_factor
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = self.prediction_window_size
        self.hidden_layer_activation = hidden_layer_activation

    def get_config(self):
        """
        Returns the config of the layer.

        A layer config is a Python dictionary (serializable) containing the configuration of a layer. The same layer can be reinstantiated later (without its trained weights) from this configuration.
        """
        return {
            "num_hidden_nodes": self.num_hidden_nodes,
            "forgetting_factor": self.forgetting_factor,
            "prediction_window_size": self.prediction_window_size,
            "hidden_layer_activation": self.hidden_layer_activation,
        }

    def build(self, input_shape):
        """
        Builds the model and operations needed for training.

        Args:
            input_shape: Single tuple, TensorShape, or list/dict of shapes, where shapes are tuples, integers, or TensorShapes.
        """

        n_input_nodes = input_shape[1]

        # hidden layer
        self.hidden_layer = tf.keras.layers.Dense(
            units=self.num_hidden_nodes,
            input_shape=(n_input_nodes,),
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-1, maxval=1
            ),
            use_bias=True,
            bias_initializer=tf.keras.initializers.RandomUniform(
                minval=-1, maxval=1
            ),
            activation=self.hidden_layer_activation,
            dtype=tf.float32,
            name="hidden_layer",
        )

        self.__p = tf.Variable(
            tf.multiply(
                tf.ones(shape=[self.num_hidden_nodes, self.num_hidden_nodes]),
                0.1,
            ),
            dtype=tf.float32,
            name="p",
        )

        # hidden to output layer connection
        self.__beta = tf.Variable(
            tf.zeros(shape=[self.num_hidden_nodes, self.num_output_nodes]),
            dtype=tf.float32,
            name="beta",
        )

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        """
        Calls the model on new inputs and returns the outputs as tensors.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            training: Boolean or boolean scalar tensor, indicating whether to run
              the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        Returns:
            A tensor if there is a single output, or a list of tensors if there are more than one outputs.
        """
        out = self.hidden_layer(inputs)
        if training is not None:
            out = tf.matmul(out, self.__beta)

        return out

    def test_step(self, data):
        """
        Custom logic for one test step.

        Args:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values of the `Model`'s metrics are returned.
        """

        # Unpack the data
        data = data_adapter.expand_1d(data)
        (
            features,
            targets,
            _,
        ) = data_adapter.unpack_x_y_sample_weight(data)

        # Compute predictions
        predictions = self(features, training=False)

        # compute loss value
        if self.compiled_loss:
            self.compiled_loss(targets, predictions)

        # Update the metrics.
        if self.compiled_metrics:
            self.compiled_metrics.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        """
        Custom logic for one training step.

        Args:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values of the `Model`'s metrics are returned.
        """

        data = data_adapter.expand_1d(data)

        # OS_ELM being SLFN (single hidden layer feedforward neural network), does not use a loss fn and hence sample_weights are ignored.
        features, targets, _ = data_adapter.unpack_x_y_sample_weight(data)

        H = self(features, training=None)

        ffi = 1.0 / self.forgetting_factor
        HT = tf.transpose(H)
        batch_size = tf.shape(H)[0]
        I = tf.eye(batch_size)
        Hp = tf.matmul(H, ffi * self.__p)
        HpHT = tf.matmul(Hp, HT)
        temp = tf.linalg.pinv(I + HpHT)
        pHT = tf.matmul(ffi * self.__p, HT)
        self.__p.assign(
            tf.subtract(ffi * self.__p, tf.matmul(tf.matmul(pHT, temp), Hp))
        )
        pHT = tf.matmul(self.__p, HT)
        Hbeta = tf.matmul(H, self.__beta)
        self.__beta.assign(
            self.__beta + tf.matmul(pHT, tf.subtract(targets, Hbeta))
        )

        # use __beta to make final prediction
        predictions = tf.matmul(H, self.__beta)

        # compute loss value
        if self.compiled_loss:
            self.compiled_loss(targets, predictions)

        # Update the metrics.
        if self.compiled_metrics:
            self.compiled_metrics.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}


class OSELM(KerasRegressor, KerasSerializationMixin):
    """
    Keras wrapper for The Online Sequential Extreme Learning Machine (OSELM).

    The Online Sequential Extreme Learning Machine (OSELM) is an online sequential learning algorithm for single hidden layer feed forward neural networks that learns the train data one-by-one or chunk-by-chunk without retraining all the historic data. It gives better generalization performance at very fast learning speed.

    OSELM being SLFN (single hidden layer feedforward neural network), does not use a loss fn to calculate gradients. It has no other control parameters for users to mannually tuning.

    Parameters
    ----------

        optimizer : Union[str, tf.keras.optimizers.Optimizer, Type[tf.keras.optimizers.Optimizer]], default "adam"
            This can be a string for Keras' built in optimizersan instance of tf.keras.optimizers.Optimizer or a class inheriting from tf.keras.optimizers.Optimizer. Only strings and classes support parameter routing.

        loss : Union[Union[str, tf.keras.losses.Loss, Type[tf.keras.losses.Loss], Callable], None], default="mse"
            The loss function to use for training. This can be a string for Keras' built in losses, an instance of tf.keras.losses.Loss or a class inheriting from tf.keras.losses.Loss .

        metrics : List[str], default=None
            list of metrics to be evaluated.

        num_hidden_nodes : int, default=100
            number of neurons to use in the hidden layer.

        hidden_layer_activation : str, default=sigmoid
            Activation function to apply on the hidden layer - can use any tf.keras.activation function name.

        prediction_window_size: int, default=1
            number of timeseries steps to predict in the future.

        forgetting_factor: float, default=0.9
            The forgetting factor can be set in order to forget the old observations quickly and track the new data tightly.

        epochs: int, default=1
            Number of training steps.

        verbose: int default=0
            To display/supress logs.
    """

    def __init__(
        self,
        loss="mse",
        optimizer=tf.keras.optimizers.Adam,
        metrics=None,
        epochs=1,
        verbose=0,
        num_hidden_nodes=25,
        hidden_layer_activation=tf.nn.sigmoid,
        prediction_window_size=1,
        forgetting_factor=0.5,
        **kwargs,
    ) -> None:
        super(OSELM, self).__init__(
            _Model(
                num_hidden_nodes,
                hidden_layer_activation,
                prediction_window_size,
                forgetting_factor,
            ),
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            epochs=epochs,
            verbose=verbose,
            **kwargs,
        )
        self.prediction_window_size = prediction_window_size

    def _ensure_compiled_model(self) -> None:
        super()._ensure_compiled_model()
        self.model_.outputs = [1] * self.prediction_window_size
