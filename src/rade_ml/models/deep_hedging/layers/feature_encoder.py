"""
Gated Residual Network (GRN) for per-timestep market feature encoding.

The GRN processes raw market observations (spot price, implied vol, moneyness,
time-to-expiry, current hedge position) into a fixed-dimensional embedding at
each timestep.  The gating mechanism allows the network to adaptively suppress
irrelevant features -- providing a degree of built-in feature selection.

Architecture:
    primary = ELU(Dense(x)) -> Dense(primary)
    gate    = sigmoid(Dense([primary, skip]))
    output  = LayerNorm(gate * primary + (1 - gate) * skip_proj)
"""
import logging
import tensorflow as tf
from typing import Any, Dict

logger = logging.getLogger(__name__)


class GatedResidualNetwork(tf.keras.layers.Layer):
    """
    Gated Residual Network for adaptive feature encoding.

    Applies a two-layer dense transform with ELU non-linearity, then uses a
    learned sigmoid gate to blend the transformed output with a skip connection.
    Layer normalisation stabilises training.

    This is the same GRN architecture used in Temporal Fusion Transformers,
    adapted here for per-timestep market state encoding in the hedging rollout.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(**kwargs)

        self.layer_config = layer_config
        self.units: int = layer_config.get("units", 64)
        self.dropout_rate: float = layer_config.get("dropout_rate", 0.0)
        self.activation: str = layer_config.get("activation", "elu")
        self.kernel_initializer: str = layer_config.get("kernel_initializer", "glorot_uniform")
        self.bias_initializer: str = layer_config.get("bias_initializer", "zeros")

        self.dense_primary = tf.keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=f"{self.name}_dense_primary",
        )
        self.dense_hidden = tf.keras.layers.Dense(
            units=self.units,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=f"{self.name}_dense_hidden",
        )
        self.gate_dense = tf.keras.layers.Dense(
            units=self.units,
            activation="sigmoid",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=f"{self.name}_gate",
        )
        self.skip_proj = tf.keras.layers.Dense(
            units=self.units,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name=f"{self.name}_skip_proj",
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name=f"{self.name}_ln"
        )
        self.dropout = (
            tf.keras.layers.Dropout(rate=self.dropout_rate, name=f"{self.name}_dropout")
            if self.dropout_rate > 0.0
            else None
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.

        :param inputs: market features [*, feature_dim]
        :param training: whether in training mode.
        :return: encoded features [*, units]
        """
        skip = self.skip_proj(inputs)

        x = self.dense_primary(inputs)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        x = self.dense_hidden(x)

        gate = self.gate_dense(tf.concat([x, skip], axis=-1))
        output = gate * x + (1.0 - gate) * skip
        return self.layer_norm(output)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["layer_config"] = self.layer_config
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GatedResidualNetwork":
        return cls(**config)
