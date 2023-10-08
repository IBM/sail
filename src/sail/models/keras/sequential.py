import tensorflow as tf
from sail.models.keras import SAILKerasClassifier
from typing import List


class _Model:
    def __init__(
        self,
        num_hidden_nodes: List[int] = [100],
        hidden_layer_activation: List[str] = ["relu"],
        num_output_nodes: int = 1,
        outer_layer_activation: str = "sigmoid",
    ):
        self.num_hidden_nodes = num_hidden_nodes
        self.hidden_layer_activation = hidden_layer_activation
        self.num_output_nodes = num_output_nodes
        self.outer_layer_activation = outer_layer_activation

    def create_instance(self):
        model = tf.keras.models.Sequential()

        assert isinstance(self.num_hidden_nodes, list) and isinstance(
            self.hidden_layer_activation, list
        ), "num_hidden_nodes and hidden_layer_activation must be of the type List[str]."

        for num_hidden_nodes, hidden_layer_activation in zip(
            self.num_hidden_nodes, self.hidden_layer_activation
        ):
            model.add(
                tf.keras.layers.Dense(
                    num_hidden_nodes, activation=hidden_layer_activation
                )
            )

        model.add(
            tf.keras.layers.Dense(
                self.num_output_nodes, activation=self.outer_layer_activation
            )
        )
        return model


class KerasSequentialClassifier(SAILKerasClassifier):
    def __init__(
        self,
        num_hidden_nodes: int = 8,
        hidden_layer_activation: str = "relu",
        num_output_nodes: int = 1,
        outer_layer_activation: str = "sigmoid",
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam,
        metrics=["accuracy"],
        epochs=1,
        verbose=0,
        **kwargs,
    ) -> None:
        super(KerasSequentialClassifier, self).__init__(
            model=_Model(
                num_hidden_nodes,
                hidden_layer_activation,
                num_output_nodes,
                outer_layer_activation,
            ).create_instance(),
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            epochs=epochs,
            verbose=verbose,
        )
        self.num_hidden_nodes = num_hidden_nodes
        self.hidden_layer_activation = hidden_layer_activation
        self.num_output_nodes = num_output_nodes
        self.outer_layer_activation = outer_layer_activation
