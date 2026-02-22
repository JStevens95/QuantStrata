"""
Recurrent hedging policy network.

At each timestep of the hedging rollout the policy observes an encoded market
state embedding and produces hedge ratios (positions in hedging instruments).

Architecture:
    encoded_feature  -->  GRU cell  -->  Dense head  -->  hedge_ratio
                          ^    |
                          |    v
                     hidden_state (carried across timesteps)

The policy is called step-by-step (one timestep at a time) by StrategyRollout,
so it exposes a ``step()`` method that takes a single-timestep input and the
previous hidden state.
"""
import logging
import tensorflow as tf
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class HedgingPolicy(tf.keras.layers.Layer):
    """
    GRU-based hedging policy that maps encoded market features to hedge ratios.

    Maintains recurrent hidden state across the rollout and outputs one hedge
    ratio per hedging instrument at each timestep.
    """

    def __init__(self, layer_config: Dict[str, Any], num_instruments: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.layer_config = layer_config
        self.num_instruments = num_instruments
        self.rnn_type: str = layer_config.get("rnn_type", "gru")
        self.rnn_units: int = layer_config.get("rnn_units", 128)
        self.rnn_layers: int = layer_config.get("rnn_layers", 2)
        self.dropout_rate: float = layer_config.get("dropout_rate", 0.0)
        self.output_activation: Optional[str] = layer_config.get("output_activation", None)
        self.kernel_initializer: str = layer_config.get("kernel_initializer", "glorot_uniform")
        self.recurrent_initializer: str = layer_config.get("recurrent_initializer", "orthogonal")
        self.bias_initializer: str = layer_config.get("bias_initializer", "zeros")

        self.gru_cells = []
        for i in range(self.rnn_layers):
            if self.rnn_type.lower() == "gru":
                cell = tf.keras.layers.GRUCell(
                    units=self.rnn_units,
                    kernel_initializer=self.kernel_initializer,
                    recurrent_initializer=self.recurrent_initializer,
                    bias_initializer=self.bias_initializer,
                    dropout=self.dropout_rate,
                    name=f"{self.name}_gru_cell_{i}",
                )
            elif self.rnn_type.lower() == "lstm":
                cell = tf.keras.layers.LSTMCell(
                    units=self.rnn_units,
                    kernel_initializer=self.kernel_initializer,
                    recurrent_initializer=self.recurrent_initializer,
                    bias_initializer=self.bias_initializer,
                    dropout=self.dropout_rate,
                    name=f"{self.name}_lstm_cell_{i}",
                )
            else:
                raise ValueError(f"Unsupported rnn_type: {self.rnn_type}")
            self.gru_cells.append(cell)

        self.output_head = tf.keras.layers.Dense(
            units=self.num_instruments,
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=f"{self.name}_output_head",
        )

    def get_initial_state(self, batch_size: int) -> list:
        """Return zero-initialised hidden states for all RNN cells."""
        states = []
        for cell in self.gru_cells:
            states.append(cell.get_initial_state(batch_size=batch_size))
        return states

    def step(
        self,
        encoded_features: tf.Tensor,
        states: list,
        training: bool = False,
    ) -> Tuple[tf.Tensor, list]:
        """
        Single-timestep forward pass.

        :param encoded_features: encoded market state [batch, encoder_units]
        :param states: list of hidden states, one per RNN layer
        :param training: training flag
        :return: (hedge_ratios [batch, num_instruments], new_states)
        """
        x = encoded_features
        new_states = []
        for cell, state in zip(self.gru_cells, states):
            x, new_state = cell(x, state, training=training)
            new_states.append(new_state)

        hedge_ratios = self.output_head(x)
        return hedge_ratios, new_states

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Full-sequence forward pass (convenience for testing).

        :param inputs: encoded features [batch, timesteps, encoder_units]
        :param training: training flag
        :return: hedge ratios [batch, timesteps, num_instruments]
        """
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        states = self.get_initial_state(batch_size)

        outputs = tf.TensorArray(dtype=tf.float32, size=timesteps, dynamic_size=False)
        for t in tf.range(timesteps):
            step_input = inputs[:, t, :]
            hedge_ratios, states = self.step(step_input, states, training=training)
            outputs = outputs.write(t, hedge_ratios)

        return tf.transpose(outputs.stack(), perm=[1, 0, 2])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["layer_config"] = self.layer_config
        config["num_instruments"] = self.num_instruments
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HedgingPolicy":
        return cls(**config)
