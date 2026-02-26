"""
RNN block for the Hybrid GNN-RNN model: LSTM, BiLSTM, or GRU.

Compresses P&L time series [B, S, T_e] into a fixed-length temporal embedding [B, d_r].
Unlike GnnBlock, RNN cells (LSTM/GRU) have activations built in (sigmoid for gates,
tanh for cell/output)—no separate block-level activation.
"""
import logging
import tensorflow as tf
from typing import Dict, Any

try:
    from keras.saving import register_keras_serializable
except ImportError:
    register_keras_serializable = tf.keras.saving.register_keras_serializable

_REGISTER_PACKAGE = "Tranql.RadeMl"

logger = logging.getLogger(__name__)


@register_keras_serializable(package=_REGISTER_PACKAGE)
class RnnBlock(tf.keras.layers.Layer):
    """
    Stack of L recurrent layers (LSTM, BiLSTM, or GRU) for temporal P&L compression.

    Layers 1..L-1 return full sequences (return_sequences=True); layer L returns
    only the final hidden state (return_sequences=False) → fixed-length embedding.
    BiLSTM doubles the output dim (forward + backward concat).
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        """
        :param layer_config: Dict with 'general' (layers, layer_type, dropout_rate)
            and 'parameters' (units, activation, recurrent_activation, initialisers).
        """
        super().__init__(**kwargs)

        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        self.layers: int | None = None
        self.layer_type: str | None = None
        self.dropout_rate: float | None = None
        self._unpack_configuration(config=layer_config.get('general'))

        self.units: int | None = None
        self.activation: str | None = None
        self.recurrent_activation: str | None = None
        self.kernel_initializer: str | None = None
        self.recurrent_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        self.rnn_layers: list = []
        self.rnn_block: tf.keras.Sequential | None = None

    def build_rnn_layers(self) -> None:
        """
        Build L RNN layers. First L-1 return sequences (for stacking); last returns
        final hidden state only.
        """
        # Intermediate layers: return_sequences=True so next layer receives full sequence.
        for i in range(self.layers - 1):
            # create the appropriate rnn layer.
            if self.layer_type.lower() == 'lstm':
                name = f'{self.name}_{self.layer_type.lower()}_{i}'
                layer = tf.keras.layers.LSTM(
                    units=self.units, activation=self.activation, recurrent_activation=self.recurrent_activation,
                    kernel_initializer=self.kernel_initializer, recurrent_initializer=self.recurrent_initializer,
                    bias_initializer=self.bias_initializer, dropout=self.dropout_rate, return_sequences=True, name=name
                )
            elif self.layer_type.lower() == 'bilstm':
                name = f'{self.name}_{self.layer_type.lower()}_{i}'
                layer = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        units=self.units, activation=self.activation, recurrent_activation=self.recurrent_activation,
                        kernel_initializer=self.kernel_initializer, recurrent_initializer=self.recurrent_initializer,
                        bias_initializer=self.bias_initializer, dropout=self.dropout_rate, return_sequences=True,
                        name=name
                    )
                )
            elif self.layer_type.lower() == 'gru':
                name = f'{self.name}_{self.layer_type.lower()}_{i}'
                layer = tf.keras.layers.GRU(
                    units=self.units, activation=self.activation, recurrent_activation=self.recurrent_activation,
                    kernel_initializer=self.kernel_initializer, recurrent_initializer=self.recurrent_initializer,
                    bias_initializer=self.bias_initializer, dropout=self.dropout_rate, return_sequences=True, name=name
                )
            elif self.layer_type.lower() == 'dense':
                name = f'{self.name}_{self.layer_type.lower()}_{i}'
                layer = tf.keras.layers.Dense(
                    units=self.units, activation=self.activation, kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer, name=name
                )
            else:
                raise ValueError(f"Undefined layer type, got {self.layer_type.lower()}")

            self.__setattr__(name, layer)
            self.rnn_layers.append(layer)

        # Final layer: return_sequences=False → output is final hidden state [B, d_r].
        if self.layer_type.lower() == 'lstm':
            name = f'{self.name}_lstm_{self.layers}'
            final_layer = tf.keras.layers.LSTM(
                units=self.units, activation=self.activation, recurrent_activation=self.recurrent_activation,
                kernel_initializer=self.kernel_initializer, recurrent_initializer=self.recurrent_initializer,
                bias_initializer=self.bias_initializer, dropout=self.dropout_rate, return_sequences=False, name=name
            )
        elif self.layer_type.lower() == 'bilstm':
            name = f'{self.name}_bilstm_{self.layers}'
            final_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=self.units, activation=self.activation, recurrent_activation=self.recurrent_activation,
                    kernel_initializer=self.kernel_initializer, recurrent_initializer=self.recurrent_initializer,
                    bias_initializer=self.bias_initializer, dropout=self.dropout_rate, return_sequences=False,
                    name=name
                )
            )
        elif self.layer_type.lower() == 'gru':
            name = f'{self.name}_gru_{self.layers}'
            final_layer = tf.keras.layers.GRU(
                units=self.units, activation=self.activation, recurrent_activation=self.recurrent_activation,
                kernel_initializer=self.kernel_initializer, recurrent_initializer=self.recurrent_initializer,
                bias_initializer=self.bias_initializer, dropout=self.dropout_rate, return_sequences=False, name=name
            )
        else:
            raise ValueError(f"Undefined layer type, got {self.layer_type.lower()}")

        self.__setattr__(name, final_layer)
        self.rnn_layers.append(final_layer)
        self.__setattr__('rnn_layers', self.rnn_layers)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build RNN stack as a Sequential."""
        _ = input_shape
        self.build_rnn_layers()
        self.rnn_block = tf.keras.Sequential(self.rnn_layers, name=f'{self.name}_rnn_stack')
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward: P [B, S, T_e] -> r [B, d_r].

        :param inputs: P&L history [batch, sequence_length, num_trades]
        :return: Temporal embedding [batch, rnn_units] (2*units for BiLSTM)
        """
        pnl_history = inputs
        x = self.rnn_block(pnl_history, training=training)
        if training:
            tf.debugging.check_numerics(x, f"NaN or Inf in x, layer: {self.rnn_block.name}")
        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Output [B, units] or [B, 2*units] for BiLSTM."""
        pnl_history = input_shape
        if self.layer_type.lower() == 'bilstm':
            return tf.TensorShape([pnl_history[0], 2 * self.units])
        return tf.TensorShape([pnl_history[0], self.units])

    def get_config(self) -> Dict[str, Any]:
        """Serialize block for saving/loading."""
        config = super(RnnBlock, self).get_config()
        config.update({'layer_config': self.layer_config})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RnnBlock":
        """Rebuild block from serialized config."""
        return cls(**config)

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict."""
        for k, v in config.items():
            setattr(self, k, v)