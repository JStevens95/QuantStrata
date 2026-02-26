"""
GNN layers for the Hybrid GNN-RNN model: GnnBlock, GraphSage, MixedGraphSage.

Design (see ARCHITECTURE.md):
- GNN sublayers (GraphSage, MixedGraphSage) are LINEAR primitives—they perform message
  passing, aggregation, and a linear transform only. Activation is applied by the block.
- GnnBlock stacks L sublayers with LayerNorm, activation, and dropout between layers,
  plus a residual connection and final activation.
"""
import copy
import logging
import tensorflow as tf
from typing import Dict, Any, Tuple, Union

try:
    from keras.saving import register_keras_serializable
except ImportError:
    register_keras_serializable = tf.keras.saving.register_keras_serializable

_REGISTER_PACKAGE = "Tranql.RadeMl"

logger = logging.getLogger(__name__)


@register_keras_serializable(package=_REGISTER_PACKAGE)
class GnnBlock(tf.keras.layers.Layer):
    """
    Graph neural network block stacking L GNN sublayers (GraphSAGE or MixedGraphSAGE)
    with residual connections, LayerNorm, activation, and dropout.

    Flow (2-layer example):
        X -> [GNN1 (linear)] -> LN -> σ -> Dropout -> [GNN2 (linear)] -> LN -> (Z + W_proj·X) -> σ -> H

    Sublayers are linear; this block applies all activation (between layers and after residual).
    Matches PyG/DGL convention and ResNet residual formulation.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        """
        :param layer_config: Dict with keys 'general' and 'parameters'.
            general: layers, layer_type, dropout_rate, use_bias, use_residual, batch_norm
            parameters: units, activation, kernel_initializer, bias_initializer
        """
        super().__init__(**kwargs)

        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        # From 'general': number of sublayers, type (graph_sage / mixed_graph_sage), etc.
        self.layers: int = 1
        self.layer_type: str | None = None
        self.dropout_rate: float = 0.0
        self.use_bias: bool | None = None
        self.use_residual: bool | None = None
        self.batch_norm: bool = False
        self._unpack_configuration(config=layer_config.get('general'))

        # From 'parameters': hidden size, activation name, initialisers.
        self.units: int | None = None
        self.activation: str | None = None
        self.kernel_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        # Built in build() / build_gnn_layers().
        self.gnn_layers: list = []
        self.norm_layers: list = []
        self.input_projection: tf.keras.layers.Dense | None = None
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_dropout') if (
                self.dropout_rate > 0.0) else None

    def build_gnn_layers(self) -> None:
        """
        Build L GNN sublayers (GraphSage or MixedGraphSage) and optionally LayerNorm.

        We deepcopy config and set activation=None so each sublayer is linear. The block
        applies activation between layers and after the residual add.
        """
        for i in range(self.layers):
            # Copy to avoid mutating caller's config; force sublayers to be linear.
            layer_config = copy.deepcopy(self.layer_config)
            layer_config['parameters']['activation'] = None

            if self.layer_type.lower() == 'graph_sage':
                name = f'{self.name}_{self.layer_type.lower()}_{i}'
                layer = GraphSage(layer_config=layer_config, name=name)
            elif self.layer_type.lower() == 'mixed_graph_sage':
                name = f'{self.name}_{self.layer_type.lower()}_{i}'
                layer = MixedGraphSage(layer_config=layer_config, name=name)
            else:
                raise ValueError(f"Undefined layer type, got {self.layer_type.lower()}")

            # Register sublayer by name for Keras tracking; add to list for iteration.
            self.__setattr__(name, layer)
            self.gnn_layers.append(layer)

            if self.batch_norm:
                name = f'{self.name}_ln_{i}'
                layer = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name=name)
                self.__setattr__(name, layer)
                self.norm_layers.append(layer)

        self.__setattr__("gnn_layers", self.gnn_layers)
        self.__setattr__("norm_layers", self.norm_layers)

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
        """
        :param input_shape: (features_shape, adjacency_shape). features: [T, p].
        """
        _, _ = input_shape

        self.build_gnn_layers()

        # Project input to match GNN output dim for residual: W_proj : R^{p} -> R^{d_g}.
        if self.use_residual and self.input_projection is None:
            self.input_projection = tf.keras.layers.Dense(
                units=self.units, kernel_initializer=self.kernel_initializer, use_bias=False,
                name=f'{self.name}_projection'
            )
            self.__setattr__('input_projection', self.input_projection)

        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, Union[tf.Tensor, tf.SparseTensor]], training: bool = False) -> tf.Tensor:
        """
        Forward pass: X, A -> H.

        :param inputs: (features [T, p], adjacency [T, T] sparse or dense)
        :return: H [T, d_g], node embeddings
        """
        features, adjacency = inputs

        # Residual branch: project X to d_g so we can add it to GNN output.
        residual = features
        if self.use_residual:
            residual = self.input_projection(residual)

        # Stack: GNN sublayer -> [LN] -> [σ, Dropout] (except after last sublayer).
        x = features
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer((x, adjacency), training=training)  # Z^(i) = GNN(x, A)

            if self.batch_norm and i < len(self.norm_layers):
                x = self.norm_layers[i](x, training=training)

            # Between layers: activation then dropout. Last sublayer skips this.
            if i < self.layers - 1:
                x = self._activation(x)
                if hasattr(self, 'dropout') and self.dropout is not None:
                    x = self.dropout(x, training=training)

        # Residual add: H = Z^(L-1) + W_proj·X
        if self.use_residual:
            x += residual

        # Final activation: H = σ(Z + residual)
        x = self._activation(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> tf.TensorShape:
        """Output shape [T, units] where T = num_trades."""
        features_shape, _ = input_shape
        return tf.TensorShape([features_shape[0], self.units])

    def get_config(self) -> Dict[str, Any]:
        """Serialize block for saving/loading."""
        config = super(GnnBlock, self).get_config()
        config.update({'layer_config': self.layer_config})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GnnBlock":
        """Rebuild block from serialized config."""
        return cls(**config)

    def _activation(self, output: tf.Tensor) -> tf.Tensor:
        """
        Apply the configured activation (ReLU, tanh, etc.) or identity if None.
        """
        if self.activation == 'leaky_relu':
            return tf.nn.leaky_relu(output, alpha=0.2)
        elif self.activation:
            return tf.keras.activations.get(self.activation)(output)
        return output

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict (e.g. layers=2, dropout_rate=0.1)."""
        for k, v in config.items():
            setattr(self, k, v)


@register_keras_serializable(package=_REGISTER_PACKAGE)
class GraphSage(tf.keras.layers.Layer):
    """
    Inductive GraphSAGE layer: h' = W_self·h + W_neigh·AGG(h | neighbors).

    Aggregation options: mean (default), max. When used inside GnnBlock, activation
    is typically None (block applies it). Standalone use can pass activation in config.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(**kwargs)

        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        self.layers: int = 1
        self.layer_type: str | None = None
        self.dropout_rate: float = 0.0
        self.use_bias: bool | None = None
        self.aggregation_op: str = 'mean'
        self._unpack_configuration(config=layer_config.get('general'))
        self.aggregation_op = getattr(self, 'aggregator_op', self.aggregation_op)

        self.units: int | None = None
        self.activation: str | None = None
        self.kernel_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        self.dense_self = None   # W_self: transform node's own features
        self.dense_neigh = None  # W_neigh: transform aggregated neighbor features
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_dropout') if (
                self.dropout_rate > 0.0) else None


    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
        """Create W_self and W_neigh Dense layers (no activation—linear only)."""
        _, _ = input_shape

        self.dense_self = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias,
            name=f'{self.name}_weights_self'
        )
        self.dense_neigh = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias,
            name=f'{self.name}_weights_neigh'
        )

        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, Union[tf.Tensor, tf.SparseTensor]], training: bool = False) -> tf.Tensor:
        """
        Forward: h' = activation(W_self·h + W_neigh·AGG(h)).

        :param inputs: (x [T, d_in], adjacency [T, T] sparse or dense)
        :return: [T, d_out]
        """
        x, a = inputs
        num_trades = tf.shape(x)[0]
        is_sparse = isinstance(a, tf.SparseTensor)

        if getattr(self, "dropout", None) is not None:
            x = self.dropout(x, training=training)

        # --- Aggregation: compute neighbor summary per node ---
        if self.aggregation_op.lower() == 'mean':
            # Mean: for row-normalised A, A@x = mean of neighbors' features per row.
            if is_sparse:
                neigh_summary = tf.sparse.sparse_dense_matmul(a, x)
            else:
                neigh_summary = tf.matmul(a, x)
        elif self.aggregation_op.lower() == 'max':
            # Max: per-feature max over neighbors. Sparse: gather by edge (col), segment_max by row.
            if is_sparse:
                rows, cols = a.indices[:, 0], a.indices[:, 1]
                gathered = tf.gather(x, cols)
                neigh_summary = tf.math.unsorted_segment_max(
                    data=gathered, segment_ids=rows, num_segments=num_trades
                )
                # Isolated nodes get -inf from segment_max; replace with zeros.
                neigh_summary = tf.where(
                    tf.math.is_finite(neigh_summary), neigh_summary, tf.zeros_like(neigh_summary)
                )
            else:
                mask = a > 0
                idx = tf.where(mask)
                rows, cols = idx[:, 0], idx[:, 1]
                gathered = tf.gather(x, cols)
                neigh_summary = tf.math.unsorted_segment_max(gathered, rows, num_segments=num_trades)
                neigh_summary = tf.where(
                    tf.math.is_finite(neigh_summary), neigh_summary, tf.zeros_like(neigh_summary)
                )
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregation_op.lower()}...")

        # --- Linear transform and combine ---
        h_self = self.dense_self(x)
        h_neigh = self.dense_neigh(neigh_summary)
        out = h_self + h_neigh
        return tf.keras.activations.get(self.activation)(out)  # linear if activation=None

    def compute_output_shape(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> tf.TensorShape:
        """Output shape [T, units] where T = num_trades."""
        features_shape, _ = input_shape
        return tf.TensorShape([features_shape[0], self.units])

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer for saving/loading."""
        config = super(GraphSage, self).get_config()
        config.update({'layer_config': self.layer_config})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphSage":
        """Rebuild layer from serialized config."""
        return cls(**config)

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict."""
        for k, v in config.items():
            setattr(self, k, v)


@register_keras_serializable(package=_REGISTER_PACKAGE)
class MixedGraphSage(tf.keras.layers.Layer):
    """
    Inductive mixed-aggregation GraphSage: h' = activation(W·[h || mean(h_neigh) || max(h_neigh)]).

    Concatenates self, mean-aggregated neighbors, and max-aggregated neighbors (3*d_in),
    then applies a single fusion Dense. Captures both smooth (mean) and salient (max) signals.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(**kwargs)

        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        self.layers: int = 1
        self.layer_type: str | None = None
        self.dropout_rate: float = 0.0
        self.use_bias: bool | None = None
        self._unpack_configuration(config=layer_config.get('general'))

        self.units: int | None = None
        self.activation: str | None = None
        self.kernel_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        self.fusion_dense = None  # W_fuse: R^{3*d_in} -> R^{d_out}
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_dropout') if (
                self.dropout_rate > 0.0) else None

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> None:
        """Create fusion Dense: input dim = 3 * d_in (self + mean + max concatenated)."""
        _, _ = input_shape

        self.fusion_dense = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias,
            name=f'{self.name}_fusion_dense'
        )

        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, Union[tf.Tensor, tf.SparseTensor]], training: bool = False) -> tf.Tensor:
        """
        Forward: concat [h, mean(h_neigh), max(h_neigh)] -> W_fuse -> activation.

        :param inputs: (x [T, d_in], adjacency [T, T])
        :return: [T, d_out]
        """
        x, a = inputs
        num_trades = tf.shape(x)[0]
        is_sparse = isinstance(a, tf.SparseTensor)

        if getattr(self, "dropout", None) is not None:
            x = self.dropout(x, training=training)

        # --- Mean aggregation: A@x (row-normalised A => mean per node) ---
        if is_sparse:
            neigh_mean = tf.sparse.sparse_dense_matmul(a, x)
        else:
            neigh_mean = tf.matmul(a, x)

        # --- Max aggregation: per-feature max over neighbors ---
        if is_sparse:
            rows, cols = a.indices[:, 0], a.indices[:, 1]
            gathered = tf.gather(x, cols)
            neigh_max = tf.math.unsorted_segment_max(
                data=gathered, segment_ids=rows, num_segments=num_trades
            )
            neigh_max = tf.where(tf.math.is_finite(neigh_max), neigh_max, tf.zeros_like(neigh_max))
        else:
            mask = a > 0
            idx = tf.where(mask)
            rows, cols = idx[:, 0], idx[:, 1]
            gathered = tf.gather(x, cols)
            neigh_max = tf.math.unsorted_segment_max(gathered, rows, num_segments=num_trades)
            neigh_max = tf.where(tf.math.is_finite(neigh_max), neigh_max, tf.zeros_like(neigh_max))

        # --- Concatenate and fuse ---
        concat_feats = tf.concat([x, neigh_mean, neigh_max], axis=1)  # [T, 3*d_in]
        out = self.fusion_dense(concat_feats)  # [T, d_out]
        return tf.keras.activations.get(self.activation)(out)

    def compute_output_shape(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape]) -> tf.TensorShape:
        """Output shape [T, units] where T = num_trades."""
        features_shape, _ = input_shape
        return tf.TensorShape([features_shape[0], self.units])

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer for saving/loading."""
        config = super(MixedGraphSage, self).get_config()
        config.update({'layer_config': self.layer_config})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MixedGraphSage":
        """Rebuild layer from serialized config."""
        return cls(**config)

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict."""
        for k, v in config.items():
            setattr(self, k, v)
