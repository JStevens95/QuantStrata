"""

"""
from __future__ import annotations

import logging
import tensorflow as tf
from typing import Dict, Any, Tuple, Union

try:
    from keras.saving import register_keras_serializable
except ImportError:
    register_keras_serializable = tf.keras.saving.register_keras_serializable

from src.rade_ml.core.base import BaseModel

# base validation checks.
from src.rade_ml.validation.base import validate_dict_keys

# hybrid gnn-rnn model custom layers.
from src.rade_ml.models.hybrid_gnn_rnn.layers.gnn_layers import GnnBlock
from src.rade_ml.models.hybrid_gnn_rnn.layers.rnn_layers import RnnBlock
from src.rade_ml.models.hybrid_gnn_rnn.layers.fusion_layer import FusionLayer
from src.rade_ml.models.hybrid_gnn_rnn.layers.attention_layer import TargetAttentionLayer
from src.rade_ml.models.hybrid_gnn_rnn.layers.projection_layer import TargetPnlOutput

# define module level logging.
logger = logging.getLogger(__name__)

# Input keys expected by call().  The adjacency is passed as three dense
# component tensors (indices, values, dense_shape) so the tf.data pipeline
# never needs to serialize a tf.SparseTensor.  The model reconstructs the
# SparseTensor internally.
_REQUIRED_KEYS = [
    'trade_features', 'pnl_history',
    'adjacency_indices', 'adjacency_values', 'adjacency_dense_shape',
    'elementary_indices', 'target_indices',
]


@register_keras_serializable(package="Tranql.RadeMl")
class HybridGnnRnn(BaseModel):
    """
    Hybrid GNN RNN model for portfolio P&L simulation.

    Integrates GNN (trade structure & relationships) with RNN (temporal PnL modeling), enabling generalisation to new
    trades and scenarios
    """
    def __init__(self, config: Dict[str, Any], name: str = "hybrid_gnn_rnn", **kwargs: Any) -> None:
        """
        Initialize the HybridGnnRnn model instance.

        :param config: configuration containing model / sub layer parameters.
        :param name: model name (default: "hybrid_gnn_rnn")
        :param kwargs: passed to BaseModel constructor.
        """
        super().__init__(name=name, **kwargs)

        # initiate required variables.
        self.model_config: Dict[str, Any] = config
        self.kwargs: Dict[str, Any] = kwargs

        # initiate required variables.
        self.general_config: Dict[str, Any] = config.get("general")
        self.gnn_config: Dict[str, Any] = config.get("gnn_layer")
        self.rnn_config: Dict[str, Any] = config.get("rnn_layer")
        self.fusion_config: Dict[str, Any] = config.get("fusion_layer")
        self.attention_config: Dict[str, Any] = config.get("attention_layer")
        self.projection_config: Dict[str, Any] = config.get("projection_layer")

        # initiate layers to build.
        self.gnn_block = GnnBlock(layer_config=self.gnn_config, name=f'{self.name}_gnn_block')
        self.rnn_block = RnnBlock(layer_config=self.rnn_config, name=f'{self.name}_rnn_block')
        self.fusion_layer = FusionLayer(layer_config=self.fusion_config, name=f'{self.name}_fusion_layer')
        self.attention_layer = TargetAttentionLayer(layer_config=self.attention_config, name=f'{self.name}_attn_layer')
        self.projection_layer = TargetPnlOutput(layer_config=self.projection_config, name=f'{self.name}_proj_layer')

        # initiate layer normalisation
        self.gnn_block_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=f'{self.name}_gnn_block_ln')
        self.rnn_block_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=f'{self.name}_rnn_block_ln')
        self.fusion_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=f'{self.name}_fusion_layer_ln')

    def build(self, input_shape: Dict[str, tf.TensorShape]) -> None:
        """
        Build the model graph.

        :param input_shape: dictionary of input tensor shapes.
        :return:
        """
        logger.info("Building Hybrid GNN-RNN model layers.")
        validate_dict_keys(input_dict=input_shape, keys=_REQUIRED_KEYS)
        logger.info("Hybrid GNN-RNN model built sucessfully.")

    def call(
            self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass of the hybrid GNN-RNN model graph.

        The adjacency matrix is received as three dense component tensors
        (adjacency_indices, adjacency_values, adjacency_dense_shape) and
        reconstructed into a tf.SparseTensor here.  This avoids tf.data
        serialization issues while preserving sparse memory efficiency.

        :param inputs: dictionary of inputs:
            - trade_features:         [num_trades, feature_dim]
            - pnl_history:            [batch, seq_len, num_elem_trades]
            - adjacency_indices:      [nnz, 2]   (int64)
            - adjacency_values:       [nnz]      (float32)
            - adjacency_dense_shape:  [2]         (int64)
            - elementary_indices:     [num_elem_trades]
            - target_indices:         [num_target_trades]
        :param training: whether in training mode.
        :return: predicted target PnL [batch, n_targets].
        """
        validate_dict_keys(input_dict=inputs, keys=_REQUIRED_KEYS)

        adjacency = tf.sparse.reorder(tf.SparseTensor(
            indices=inputs["adjacency_indices"],
            values=inputs["adjacency_values"],
            dense_shape=inputs["adjacency_dense_shape"],
        ))

        return self.run_model(
            inputs=(
                inputs["trade_features"],
                inputs["pnl_history"],
                adjacency,
                inputs["target_indices"],
            ),
            training=training,
        )

    def run_model(
            self, inputs: Tuple[tf.Tensor, tf.Tensor, Union[tf.Tensor, tf.SparseTensor], tf.Tensor],
            training: bool = False
    ) -> tf.Tensor:
        """
        Run the Hybrid GNN-RNN architecture:
            1. GNN block
            2. RNN block
            3. Fusion layer (cross attention + gating)
            4. Target attention layer
            5. PnL projection

        :param inputs: (trade_features, pnl_history, adjacency, target_indices).
        :param training: whether in training mode.
        :return: predictions [batch, n_targets].
        """
        trade_features, elementary_pnl, adjacency, target_indices = inputs

        # 1. apply GNN block --> [num_trades, gnn_dim]
        gnn_features = self.gnn_block(inputs=(trade_features, adjacency), training=training)
        gnn_features = self.gnn_block_ln(gnn_features)

        # 2. apply RNN block --> [batch, rnn_dim]
        rnn_features = self.rnn_block(inputs=elementary_pnl, training=training)
        rnn_features = self.rnn_block_ln(rnn_features)

        # 3. apply fusion layer; combined GNN & RNN embedding --> [batch, num_trades, fusion_dim]
        fused_features = self.fusion_layer(
            inputs=(gnn_features, rnn_features, adjacency), training=training
        )
        fused_features = self.fusion_ln(fused_features)

        # 4. apply target attention layer --> [batch, num_targets, attn_dim]
        attended_features = self.attention_layer(
            inputs=(fused_features, adjacency, target_indices), training=training
        )

        # 5. apply pnl output projection --> [batch, num_targets]
        return self.projection_layer(
            inputs=(trade_features, attended_features, target_indices), training=training
        )

    @staticmethod
    def compute_output_shape(input_shape: Dict[str, tf.TensorShape]) -> tf.TensorShape:
        """
        Compute output shape of the model.
        :param input_shape: dictionary of input shapes
        :return:
        """
        pnl_history = input_shape.get("pnl_history")
        target_indices = input_shape.get("target_indices")
        return tf.TensorShape([pnl_history[0], target_indices[0]])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization (metadata from BaseModel + config)."""
        config = super().get_config()
        config["model_config"] = self.model_config
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "HybridGnnRnn":
        """Instantiate HybridGnnRnn from serialized configuration."""
        metadata = config.pop("metadata", {})
        model_config = config.pop("model_config", {})
        model = cls(config=model_config, **config, **kwargs)
        model._model_metadata.update(metadata)
        return model
