"""
Target attention: self-attention over target trades plus FFN sublayer.

Restricts to target indices only, extracts adjacency submatrix (sparse or dense),
computes multi-head self-attention masked by adjacency, then FFN with residual.
"""
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
class TargetAttentionLayer(tf.keras.layers.Layer):
    """
    Self-attention over target trades [B, n_tgt, d] masked by adjacency.

    Flow: gather target features -> extract [n_tgt,n_tgt] submatrix -> Q,K,V projection ->
    multi-head attention (sparse O(n*k) or dense O(n^2)) -> residual + LN -> FFN -> residual + LN.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        super().__init__(**kwargs)

        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        self.dropout_rate: float = 0.0
        self.num_heads: int = 1
        self.k_nbrs: int = 50
        self._unpack_configuration(config=layer_config.get('general'))

        self.units: int | None = None
        self.activation: str | None = None
        self.kernel_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        assert self.units % self.num_heads == 0, "units must be divisible by num_heads."
        self.head_units = self.units // self.num_heads
        self.units_ffn = 4 * self.units

        # Project fused features to attention space.
        self.fused_proj = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, name=f'{self.name}_fused_projection'
        )

        # Q,K,V and output projections for self-attention.
        self.q_dense = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=False,
            name=f'{self.name}_query_projection'
        )
        self.k_dense = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=False,
            name=f'{self.name}_key_projection'
        )
        self.v_dense = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=False,
            name=f'{self.name}_value_projection'
        )
        self.out_dense = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, use_bias=False, name=f'{self.name}_output_projection'
        )

        self.layer_norm = tf.keras.layers.LayerNormalization(name=f'{self.name}_layer_normalisation')
        self.attn_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_attn_dropout')

        # FFN sublayer: linear -> activation -> linear (standard Transformer pattern).
        self.ffn_dense_1 = tf.keras.layers.Dense(
            units=self.units_ffn, activation=self.activation, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, name=f'{self.name}_ffn_dense_1'
        )
        self.ffn_dense_2 = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, name=f'{self.name}_ffn_dense_2'
        )
        self.ffn_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_ffn_dropout')
        self.ffn_norm = tf.keras.layers.LayerNormalization(name=f'{self.name}_ffn_norm')

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]) -> None:
        """Input: (fused [B,T,d_f], adjacency [T,T], target_idx)."""
        _, _, _ = input_shape
        super().build(input_shape)

    def call(
            self, inputs: Tuple[tf.Tensor, Union[tf.Tensor, tf.SparseTensor], tf.Tensor], training: bool = False,
            return_attention: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
        """
        Forward: (fused, adjacency, target_idx) -> [B, n_tgt, units].

        Extracts target submatrix, runs self-attention + FFN, optionally returns attention weights.
        """
        fused_features, adjacency, target_idx = inputs

        # Restrict to target trades only.
        fused_features = tf.gather(fused_features, target_idx, axis=1)

        # Extract small [n_tgt, n_tgt] binary adjacency submatrix without
        # materializing the full [T, T] dense matrix.  We filter the sparse
        # indices to keep only entries where both row AND column are targets,
        # then remap to local [0..n_tgt-1] coordinates.
        adjacency = self._extract_target_submatrix(adjacency, target_idx)
        _, num_trades = tf.shape(fused_features)[0], tf.shape(fused_features)[1]

        fused = self.fused_proj(fused_features)
        query = self.q_dense(fused)
        key = self.k_dense(fused)
        value = self.v_dense(fused)

        query_h, key_h, value_h = (self._split_heads(x) for x in (query, key, value))
        core_out = self._core_calc(
            q=query_h, k=key_h, v=value_h, adjacency=adjacency, num_trades=num_trades, training=training,
            return_attention=return_attention
        )
        if return_attention:
            context, attn_weights, attn_extra = core_out
        else:
            context = core_out

        attn_out = self._combine_heads(context)
        attn_out = self.out_dense(attn_out)
        attn_out = self.layer_norm(fused + attn_out)  # Residual.

        # FFN sublayer with residual.
        ffn = self.ffn_dense_1(attn_out)
        ffn = self.ffn_dropout(ffn, training=training)
        ffn = self.ffn_dense_2(ffn)
        output = self.ffn_norm(attn_out + ffn)

        if return_attention:
            expl = {"target_attention": attn_weights, **attn_extra}
            return output, expl
        return output

    def compute_output_shape(
            self, input_shape: Tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]
    ) -> tf.TensorShape:
        """Output shape: [None, None, units] (batch and n_tgt may be dynamic)."""
        _, _, _ = input_shape
        return tf.TensorShape([None, None, self.units])

    def get_config(self) -> Dict[str, Any]:
        config = super(TargetAttentionLayer, self).get_config()
        config.update({
            'layer_config': self.layer_config
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TargetAttentionLayer":
        """Instantiate from serialized config."""
        return cls(**config)

    @staticmethod
    def _extract_target_submatrix(
            adjacency: Union[tf.Tensor, tf.SparseTensor], target_idx: tf.Tensor
    ) -> Union[tf.Tensor, tf.SparseTensor]:
        """
        Extract a small [n_tgt, n_tgt] adjacency submatrix.

        When ``adjacency`` is a SparseTensor (typical case), this avoids
        materializing the full [T, T] dense matrix.  Instead it:
          1. Builds a lookup table mapping global trade id -> local target id.
          2. Gathers the local ids for every sparse edge's row and column.
          3. Keeps only edges where **both** endpoints are target trades.
          4. Constructs a row-major [n_tgt, n_tgt] SparseTensor.

        The SparseTensor is returned directly so that downstream attention
        can use the O(n_tgt * k) sparse path instead of O(n_tgt^2) dense.

        Cost: O(nnz) scan — trivial vs. O(T^2).

        :param adjacency: full trade adjacency, sparse or dense [T, T].
        :param target_idx: 1-D int tensor of global target trade indices.
        :return: SparseTensor [n_tgt, n_tgt] if input is sparse, else dense
                 binary float32 tensor [n_tgt, n_tgt].
        """
        if isinstance(adjacency, tf.SparseTensor):
            n_tgt = tf.shape(target_idx)[0]
            T = adjacency.dense_shape[0]

            # Build a lookup: global_id -> local_id (or -1 if not a target).
            lookup = tf.fill([T], tf.cast(-1, target_idx.dtype))
            local_ids = tf.range(n_tgt, dtype=target_idx.dtype)
            lookup = tf.tensor_scatter_nd_update(
                lookup, tf.expand_dims(target_idx, 1), local_ids
            )

            rows = adjacency.indices[:, 0]
            cols = adjacency.indices[:, 1]
            local_rows = tf.gather(lookup, rows)
            local_cols = tf.gather(lookup, cols)

            # Keep edges where both endpoints are targets (local id >= 0).
            keep = (local_rows >= 0) & (local_cols >= 0)
            local_rows = tf.boolean_mask(local_rows, keep)
            local_cols = tf.boolean_mask(local_cols, keep)

            sub_indices = tf.stack(
                [tf.cast(local_rows, tf.int64), tf.cast(local_cols, tf.int64)], axis=1
            )
            sub_values = tf.ones([tf.shape(sub_indices)[0]], dtype=tf.float32)
            sub_shape = tf.cast([n_tgt, n_tgt], tf.int64)
            return tf.sparse.reorder(
                tf.SparseTensor(sub_indices, sub_values, sub_shape)
            )
        else:
            adjacency = tf.cast(adjacency > 0, tf.float32)
            return tf.gather(tf.gather(adjacency, target_idx, axis=0), target_idx, axis=1)

    def _core_calc(
            self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
            adjacency: Union[tf.Tensor, tf.SparseTensor], num_trades, training: bool = False,
            return_attention: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]]:
        """
        Core attention — dispatches to O(n_tgt * k) sparse neighborhood
        attention when the target submatrix is a SparseTensor, or O(n_tgt^2)
        dense attention otherwise.

        :param q: query tensor  [B, h, n_tgt, d_h]
        :param k: key tensor    [B, h, n_tgt, d_h]
        :param v: value tensor  [B, h, n_tgt, d_h]
        :param adjacency: target submatrix (sparse or dense) [n_tgt, n_tgt]
        :param num_trades: scalar n_tgt
        :param training: whether in training mode
        :param return_attention: if True, return (context, weights, extra_dict).
        :return: context tensor [B, h, n_tgt, d_h]
        """
        if isinstance(adjacency, tf.SparseTensor):
            return self._sparse_target_attention(q, k, v, adjacency, num_trades, training, return_attention)
        return self._dense_target_attention(q, k, v, adjacency, num_trades, training, return_attention)

    def _dense_target_attention(
            self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
            adjacency: tf.Tensor, num_trades, training: bool = False,
            return_attention: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]]:
        """Standard O(n_tgt^2) attention: softmax(masked QK^T/√d) @ V."""
        scores = tf.matmul(q, k, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.head_units, scores.dtype))

        mask = tf.cast(adjacency > 0, scores.dtype)
        mask = tf.reshape(mask, [1, 1, num_trades, num_trades])

        very_neg = tf.cast(-1e9, scores.dtype)
        scores_masked = tf.where(mask > 0, scores, very_neg)
        weights = tf.nn.softmax(scores_masked, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        context = tf.matmul(weights, v)
        if return_attention:
            return context, weights, {}
        return context

    def _sparse_target_attention(
            self, q: tf.Tensor, k_proj: tf.Tensor, v: tf.Tensor,
            adjacency: tf.SparseTensor, num_trades, training: bool = False,
            return_attention: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]]:
        """
        O(n_tgt * k) sparse neighborhood attention over target trades.

        Instead of materializing a [n_tgt, n_tgt] score matrix, each target
        trade only attends to its k neighbors from the target adjacency
        subgraph.  Memory goes from O(B * h * n_tgt^2) to O(B * h * n_tgt * k).

        :param q: query  [B, h, n_tgt, d_h]
        :param k_proj: key    [B, h, n_tgt, d_h]
        :param v: value  [B, h, n_tgt, d_h]
        :param adjacency: sparse target submatrix [n_tgt, n_tgt] (row-major)
        :param num_trades: scalar n_tgt
        :param training: whether in training mode
        :return: context [B, h, n_tgt, d_h]
        """
        rows = adjacency.indices[:, 0]
        cols = adjacency.indices[:, 1]
        num_trades_i64 = tf.cast(num_trades, tf.int64)

        # Per-row neighbor counts — uniform for a k-NN graph, but handles
        # variable-degree gracefully.
        row_counts = tf.math.unsorted_segment_sum(
            tf.ones_like(rows, dtype=tf.int32), tf.cast(rows, tf.int32), num_trades
        )
        k = tf.reduce_max(row_counts)

        # Build padded [n_tgt, k] neighbor index array from the sparse
        # structure.  from_value_rowids requires sorted (non-decreasing)
        # row ids — guaranteed because the SparseTensor was reordered to
        # row-major by _extract_target_submatrix.
        nbr_ragged = tf.RaggedTensor.from_value_rowids(cols, rows, nrows=num_trades_i64)
        nbr_idx = tf.cast(nbr_ragged.to_tensor(default_value=tf.constant(0, cols.dtype)), tf.int32)
        nbr_mask = tf.sequence_mask(row_counts, maxlen=k, dtype=q.dtype)

        # Gather neighbor keys and values along the n_tgt axis.
        # tf.gather(params=[B,h,n_tgt,d_h], indices=[n_tgt,k], axis=2) -> [B,h,n_tgt,k,d_h]
        k_nbr = tf.gather(k_proj, nbr_idx, axis=2)
        v_nbr = tf.gather(v, nbr_idx, axis=2)

        # Scores: dot(query, neighbor_key) -> [B, h, n_tgt, k]
        q_exp = tf.expand_dims(q, axis=3)                              # [B, h, n_tgt, 1, d_h]
        scores = tf.reduce_sum(q_exp * k_nbr, axis=-1)                 # [B, h, n_tgt, k]
        scores /= tf.math.sqrt(tf.cast(self.head_units, scores.dtype))

        # Mask padded neighbor positions with -1e9 so softmax drives them to ~0.
        mask_bcast = tf.reshape(nbr_mask, [1, 1, num_trades, k])       # [1, 1, n_tgt, k]
        very_neg = tf.cast(-1e9, scores.dtype)
        scores = tf.where(mask_bcast > 0, scores, very_neg)

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        # Weighted context: [B, h, n_tgt, d_h]
        context = tf.reduce_sum(tf.expand_dims(weights, -1) * v_nbr, axis=3)
        if return_attention:
            return context, weights, {"target_neighbor_indices": nbr_idx}
        return context

    def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Merge heads: [B, h, T, d_h] -> [B, T, units]."""
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        return tf.reshape(x, [b, t, self.units])

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Split last dim into heads: [B, T, units] -> [B, h, T, d_h]."""
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.num_heads, self.head_units])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict."""
        for k, v in config.items():
            setattr(self, k, v)
