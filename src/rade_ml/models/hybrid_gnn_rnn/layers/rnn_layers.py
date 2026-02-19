import logging
import tensorflow as tf
from typing import Dict, Any

# define logging at module level.
logger = logging.getLogger(__name__)


class RnnBlock(tf.keras.layers.Layer):
    """
    Recurrent neural network block with multiple configurable layers.

    This block implements a stack of RNN layers: [lstm, bilstm, gru]
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        """
        Initiate recurrent neural network block.

        :param layer_config: dictionary containing general layer configuration & parameters.
        """
        # init call to super class
        super().__init__(**kwargs)

        # initiate required variables.
        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        # unpack layer configuration --> general.
        self.layers: int | None = None
        self.layer_type: str | None = None
        self.dropout_rate: float | None = None
        self._unpack_configuration(config=layer_config.get('general'))

        # unpack layer configuration --> parameters.
        self.units: int | None = None
        self.activation: str | None = None
        self.recurrent_activation: str | None = None
        self.kernel_initializer: str | None = None
        self.recurrent_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        # initiate variables to build.
        self.rnn_layers: list = []
        self.rnn_block: tf.keras.Sequential | None = None

    def build_rnn_layers(self) -> None:
        """
        Build rnn sequential layers.

        Supported layers are as follows:
             - LSTM
             - BiLSTM
             - GRU
        :return:
        """

        # loop through specified layer(s) / type(s) to build rnn block.
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

            # append rnn layers to list.
            self.__setattr__(name, layer)
            self.rnn_layers.append(layer)

        # build final rnn layer. --> return sequence = False
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

        # append final layer to list.
        self.__setattr__(name, final_layer)
        self.rnn_layers.append(final_layer)

        # define weight tracking for layer list.
        self.__setattr__('rnn_layers', self.rnn_layers)

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the block with given input shape.

        :param input_shape: pnl_timeseries [batch, sequence, num_trades]
        :return:
        """
        # extract input shape
        _ = input_shape

        # build dynamic rnn layers.
        self.build_rnn_layers()

        # create a sequential from already-configured layer instances.
        self.rnn_block = tf.keras.Sequential(self.rnn_layers, name=f'{self.name}_rnn_stack')

        # update layer built flag.
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass of RnnBlock

        :param inputs: trade pnl history [batch, sequence, num_trades]
        :param training: whether in training mode.
        :return:
        """
        # extract inputs.
        pnl_history = inputs

        # input pnl timeseries tensor into rnn layers.
        x = self.rnn_block(pnl_history, training=training)
        tf.debugging.check_numerics(x, f"NaN or Inf in x, layer: {self.rnn_block.name}")

        # expected output shape: [batch, rnn_units]
        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """
        Compute output shape of the layer.

        :param input_shape:
        :return:
        """
        # extract input shapes.
        pnl_history = input_shape
        if self.layer_type.lower() == 'bilstm':
            return tf.TensorShape([pnl_history[0], 2 * self.units])
        else:
            return tf.TensorShape([pnl_history[0], self.units])

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for serializing the layer.

        :return:
        """
        config = super(RnnBlock, self).get_config()
        config.update({
            'layer_config': self.layer_config
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RnnBlock":
        """
        Instantiates the RnnBlock from its config.

        :param config:
        :return:
        """
        return cls(**config)

    def _unpack_configuration(
            self, config: Dict[str, Any]
    ) -> None:
        """
        Unpack configuration elements into separate variables.
        :param config: dictionary configuration.
        :return:
        """
        for k, v in config.items():
            setattr(self, k, v)