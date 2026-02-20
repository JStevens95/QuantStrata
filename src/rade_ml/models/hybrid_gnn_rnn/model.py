"""

"""
from __future__ import annotations

import logging
import tensorflow as tf
from typing import Dict, Any, Union, Tuple

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

        :param input_shape: dictionary of input tensor shape.
            - trade features: [num_trades, feature_dim]
            - pnl history: [batch, num_elem_trades]
            - adjacency matrix: [num_trades, num_trades]
            - elementary indices: [num_elem_trades]
            - target indices: [num_target_trades]
        :return:
        """
        logger.info("Building Hybrid GNN-RNN model layers.")

        # ensure required keys and shapes are present
        validate_dict_keys(
            input_dict=input_shape,
            keys=['trade_features', 'pnl_history', 'adjacency_matrix', 'elementary_indices', 'target_indices'],
        )

        # update model build flag.
        logger.info("Hybrid GNN-RNN model built sucessfully.")

    def call(self, inputs: Dict[str, Union[tf.Tensor, tf.SparseTensor]], training: bool = False) -> tf.Tensor:
        """
        Forward pass of the hybrid GNN-RNN model graph.

        This method orchestrates the entire computational flow through the model:
            1. Feature extraction and validation.
            2. GNN processing to capture trade relationships.
            3. RNN processing to capture temporal pnl dependencies.
            4. Feature fusion to combine structural and temporal embeddings.
            5. Attention mechanism to focus on relevant target trades.
            6. Final projection to generate PnL predictions.

        :param inputs: dictionary of inputs:
            - trade features: [num_trades, feature_dim]
            - pnl history: [batch, num_elem_trades]
            - adjacency matrix: [num_trades, num_trades]
            - elementary indices: [num_elem_trades]
            - target indices: [num_target_trades]
        :param training: whether in training mode.
        :return:
        """
        # validate input keys.
        validate_dict_keys(
            input_dict=inputs,
            keys=['trade_features', 'pnl_history', 'adjacency_matrix', 'elementary_indices', 'target_indices'],
        )

        # extract tensors from inputs
        trade_Features = inputs.get("trade_features")
        pnl_history = inputs.get("pnl_history")
        adjacency = inputs.get("adjacency_matrix")
        target_indices = inputs.get("target_indices")

        # run hybrid model logic.
        output = self.run_model(inputs=(trade_Features, pnl_history, adjacency, target_indices), training=training)
        return output

    def run_model(
            self, inputs: Tuple[tf.Tensor, tf.Tensor, Union[tf.Tensor, tf.SparseTensor], tf.Tensor],
            training: bool = False
    ) -> tf.Tensor:
        """
        Running Hybrid GNN-RNN architecture.
            1. run gnn block
            2. run rnn block
            3. run fusion layer - using cross attention and gating mechanism and similarity weighting.
            4. run target attention layer - using similarity weighting.
            5. run pnl output projection

        :param inputs: tuple of inputs:
            - trade features: [num_trades, feature_dim]
            - pnl history: [batch, num_elem_trades]
            - adjacency matrix: [num_trades, num_trades]
            - target indices: [num_target_trades]
        :param training: whether in training mode.
        :return:
        """
        # extract inputs
        trade_features, elementary_pnl, adjacency, target_indices = inputs

        # 1. apply GNN block --> [num_trades, gnn_dim]
        gnn_features = self.gnn_block(inputs=(trade_features, adjacency), training=training)
        gnn_features = self.gnn_block_ln(gnn_features)

        # 2. apply RNN block --> [batch, rnn_dim]
        rnn_features = self.rnn_block(inputs=elementary_pnl, training=training)
        rnn_features = self.rnn_block_ln(rnn_features)

        # 3. apply fusion layer; combined GNN & RNN embedding --> [batch, num_trades, fusion_dim]
        fused_features = self.fusion_layer(inputs=(gnn_features, rnn_features, adjacency), training=training)
        fused_features = self.fusion_ln(fused_features)

        # 4. apply target attention layer --> [batch, num_targets, attn_dim]
        attended_features = self.attention_layer(inputs=(fused_features, adjacency, target_indices), training=training)

        # 5. apply pnl output projection.
        outputs = self.projection_layer(inputs=(trade_features, attended_features, target_indices), training=training)
        return outputs

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
        """Instantiate HybridGnnRnn from configuration (delegates to BaseModel for metadata handling)."""
        return super().from_config(config, **kwargs)
