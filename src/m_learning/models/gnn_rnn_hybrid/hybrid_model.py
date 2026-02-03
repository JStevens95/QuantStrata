import logging
import tensorflow as tf
from typing import Dict, Any, Union, Tuple
from src.m_learning.models.gnn_rnn_hybrid.layers import (
    GnnBlock,
    RnnBlock,
    FusionLayer,
    TargetAttentionLayer,
    TargetPnlOutput,
)

# define logging at module level.
logger = logging.getLogger(__name__)


class HybridGnnRnn(tf.keras.Model):
    """
    Hybrid GNN-RNN model for PnL simulation.

    This model integrates the GNN for modelling trade structure & relationships with RNN for PnL temporal modelling,
    enabling better generalisation oto new trades with different attributes and new scenarios.
    """

    def __init__(self, model_config: Dict[str, Any], **kwargs) -> None:
        """
        Initialise the Hybrid GNN-RNN model.

        :param model_config:  dictionary configuration containing model / sub-layer parameters.
        """
        # init call to super class
        super().__init__(**kwargs)

        # initiate required variables.
        self.model_config = model_config
        self.kwargs = kwargs

        # initiate derived variables.
        self.general_config: Dict[str, Any] = model_config.get('general')
        self.gnn_config: Dict[str, Any] = model_config.get('gnn_model')
        self.rnn_config: Dict[str, Any] = model_config.get('rnn_model')
        self.fusion_config: Dict[str, Any] = model_config.get('fusion_model')
        self.attention_config: Dict[str, Any] = model_config.get('attention_model')
        self.projection_config: Dict[str, Any] = model_config.get('projection_model')

        # initiate layers to build.
        self.gnn_block = GnnBlock(layer_config=self.gnn_config, name=f'{self.name}_gnn_block')
        self.rnn_block = RnnBlock(layer_config=self.rnn_config, name=f'{self.name}_rnn_block')
        self.fusion_layer = FusionLayer(layer_config=self.fusion_config, name=f'{self.name}_fusion_layer')
        self.attention_layer = TargetAttentionLayer(layer_config=self.attention_config, name=f'{self.name}_attn_layer')
        self.pnl_projection = TargetPnlOutput(layer_config=self.projection_config, name=f'{self.name}_projection_layer')
        self.rnn_only_output: tf.keras.layers.Dense | None = None

        # initiate layer normalisations.
        self.gnn_block_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=f'{self.name}_gnn_block_ln')
        self.rnn_block_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=f'{self.name}_rnn_block_ln')
        self.fusion_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=f'{self.name}_fusion_ln')

    def build(self, input_shape: Dict[str, tf.TensorShape]) -> None:
        """

        :param input_shape: dictionary of input shapes:
             - trade features: [num_trades, feature_dim]
             - pnl_history: [batch, num_elem_trades]
             - adjacency matrix: [num_trades, num_trades]
             - elementary indices: [num_elem_trades]
             - target indices: [num_targ_trades]
        :return:
        """
        logger.info("Building Hybrid GNN-RNN model layers...")

        # ensure required keys and shapes are present
        required_keys = ['trade_features', 'pnl_history', 'adjacency_matrix', 'elementary_indices', 'target_indices']
        missing_keys = [k for k in required_keys if k not in input_shape]
        if missing_keys:
            raise ValueError(f"Missing required shapes for keys: {missing_keys}")

        # extract input shapes
        features_shape = input_shape['trade_features']
        target_indices = input_shape['target_indices']

        # build output layer for rnn only mode.
        architecture = self.general_config.get('architecture', 'default').lower()
        if architecture == 'rnn_only':
            self.rnn_only_output = tf.keras.layers.Dense(
                units=target_indices[0], activation=None, name=f'{self.name}_rnn_only_output'
            )

        # update model built flag
        logger.info("Hybrid GNN-RNN model built successfully...")

    def call(self, inputs: Dict[str, Union[tf.Tensor, tf.SparseTensor]], training: bool = False) -> tf.Tensor:
        """
        Forward pass of the hybrid model.

        This method orchestrates the entire computational flow through the model:
            1. Feature extraction and validation.
            2. GNN processing to capture trade relationships
            3. RNN processing for temporal dependencies.
            4, Feature fusion to combine structural and temporal embedding.
            5. Attention mechanism to focus on relevant target trades.
            6. Final projection to generate PnL predictions.

        :param inputs: dictionary of inputs:
             - trade features: [num_trades, feature_dim]
             - pnl_history: [batch, sequence, num_elem_trades]
             - adjacency matrix: [num_trades, num_trades]
             - elementary indices: [num_elem_trades]
             - target indices: [num_targ_trades]
        :param training: whether in training mode.
        :return:
        """
        # extract tensors from inputs
        trade_features = inputs['trade_features']
        pnl_history = inputs['pnl_history']
        adjacency = inputs['adjacency_matrix']
        target_indices = tf.cast(inputs['target_indices'], dtype=tf.int32)

        # When tf.data.Dataset yields batched dicts, trade_features/adjacency/target_indices
        # have an extra batch dim (same graph for all samples). Squeeze to rank-2/rank-1 for the model.
        if len(trade_features.shape) == 3:
            trade_features = trade_features[0]
        if len(adjacency.shape) == 3:
            adjacency = adjacency[0]
        if len(target_indices.shape) == 2:
            target_indices = target_indices[0]
        elementary_indices = inputs.get('elementary_indices')
        if elementary_indices is not None and len(elementary_indices.shape) == 2:
            elementary_indices = elementary_indices[0]

        model_inputs = {
            'trade_features': trade_features,
            'pnl_history': pnl_history,
            'adjacency_matrix': adjacency,
            'target_indices': target_indices,
        }
        if elementary_indices is not None:
            model_inputs['elementary_indices'] = elementary_indices
        self._validate_inputs(model_inputs)

        # run different orchestration depending on mode.
        architecture = self.general_config.get('architecture', 'default').lower()
        if architecture == 'rnn_only':
            output = self.run_rnn_model(inputs=pnl_history, training=training)
            return output
        elif architecture == 'default':
            output = self.run_default_model(
                inputs=(trade_features, pnl_history, adjacency, target_indices),
                training=training,
            )
            return output
        else:
            raise ValueError(f"Unsupported mode {architecture}; supported modes ['default', 'rnn_only']")

    def run_rnn_model(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Run RNN model only.
        :param inputs:
        :param training:
        :return:
        """
        # extract inputs.
        pnl_history = inputs

        # 1. apply RNN Block --> [batch, rnn_units]
        rnn_embedding = self.rnn_block(inputs=pnl_history, training=training)

        # 2. project to pnl predictions --> [batch, num_targ_trades]
        output = self.rnn_only_output(rnn_embedding)
        return output

    def run_default_model(
            self, inputs: Tuple[tf.Tensor, tf.Tensor, Union[tf.Tensor, tf.SparseTensor], tf.Tensor],
            training: bool = False
    ) -> tf.Tensor:
        """
        Running default Hybrid GNN-RNN architecture.
            1. run gnn block
            2. run rnn block
            3. run fusion layer - using cross attention and gating mechanism and similarity weighting.
            4. run target attention layer - using similarity weighting.
            5. run pnl output projection

        :param inputs: tuple of inputs:
             - trade features: [num_trades, feature_dim]
             - pnl_history: [batch, num_elem_trades]
             - adjacency matrix: [num_trades, num_trades]
             - target indices: [num_targ_trades]
        :param training: whether in training mode.
        :return:
        """
        # extract inputs.
        trade_features, elementary_pnl, adjacency, target_indices = inputs

        # 1. apply GNN block --> [num_trades, gnn_dim]
        gnn_features = self.gnn_block(inputs=(trade_features, adjacency), training=training)
        gnn_features = self.gnn_block_ln(gnn_features)

        # 2. apply RNN block --> [batch, rnn_dim]
        rnn_features = self.rnn_block(inputs=elementary_pnl, training=training)
        rnn_features = self.rnn_block_ln(rnn_features)

        # 3. apply fusion layer; combined GNN and RNN embedding --> [batch, num_trades, fusion_dim]
        fused_features = self.fusion_layer(inputs=(gnn_features, rnn_features, adjacency), training=training)
        fused_features = self.fusion_ln(fused_features)

        # 4. apply target attention layer --> [batch, num_targets, attn_dim]
        attended_features = self.attention_layer(inputs=(fused_features, adjacency, target_indices), training=training)

        # 5. apply pnl output projection
        outputs = self.pnl_projection(inputs=(trade_features, attended_features, target_indices), training=training)
        return outputs

    @staticmethod
    def compute_output_shape(input_shape: Dict[str, tf.TensorShape]) -> tf.TensorShape:
        """
        Compute output shape of the layer.
        :param input_shape: dictionary of input shapes:
             - trade features: [num_trades, feature_dim]
             - pnl_history: [batch, num_elem_trades]
             - adjacency matrix: [num_trades, num_trades]
             - elementary indices: [num_elem_trades]
             - target indices: [num_targ_trades]
        :return:
        """
        pnl_history = input_shape['pnl_history']
        target_indices = input_shape['target_indices']
        return tf.TensorShape([pnl_history[0], target_indices[0]])

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for serializing the layer.
        :return:
        """
        config = super(HybridGnnRnn, self).get_config()
        config.update({
            'model_config': self.model_config
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HybridGnnRnn":
        """
        Instantiates the HybridGnnRnn from its config.

        :param config:
        :return:
        """
        return cls(**config)

    @staticmethod
    def _validate_inputs(inputs: Dict[str, Union[tf.Tensor, tf.SparseTensor]]) -> None:
        required = ["trade_features", "pnl_history", "adjacency_matrix", "target_indices"]
        missing = [k for k in required if k not in inputs]
        if missing:
            raise ValueError(f"Missing required input keys: {missing}")

        # Basic rank checks
        tf.debugging.assert_rank(inputs["trade_features"], 2, message="trade_features must be [T, P]")
        tf.debugging.assert_rank(inputs["pnl_history"], 3, message="pnl_history must be [B, S, N_elem]")
        adj = inputs["adjacency_matrix"]
        if isinstance(adj, tf.SparseTensor):
            tf.debugging.assert_shapes([(adj.dense_shape, (2,))], message="adjacency_matrix dense_shape must be 2-D")
        else:
            tf.debugging.assert_rank(adj, 2, message="adjacency_matrix must be [T, T]")
        tf.debugging.assert_rank(inputs["target_indices"], 1, message="target_indices must be [N_tgt]")
