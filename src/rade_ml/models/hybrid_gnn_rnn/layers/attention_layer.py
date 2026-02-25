import logging
import tensorflow as tf
from typing import Dict, Any, Tuple, Union

try:
    from keras.saving import register_keras_serializable
except ImportError:
    register_keras_serializable = tf.keras.saving.register_keras_serializable

_REGISTER_PACKAGE = "Tranql.RadeMl"

# define logging at module level
logger = logging.getLogger(__name__)


@register_keras_serializable(package=_REGISTER_PACKAGE)
class TargetAttentionLayer(tf.keras.layers.Layer):
    """
    Inter-trade self attention + position wise feed-forward with multiplicative similarity re-weight.

    - Self attention lets each trade look at all other trades' fused features, weighting them by relevance.
    - Similarity re-weighting enforces known attributes relationships from adjacency matrix.
    - Residual connection ensures the original fused signal isn't lost if the attention module isn't helpful.
    - Feed-forward MLP adds extra non-linearity so the layer can learn richer per trade transformations.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        """
        Initialise the TargetAttentionLayer.

        :param layer_config: dictionary containing general layer configuration & parameters.
        """
        # init call to super class
        super().__init__(**kwargs)

        # initiate required variables.
        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        # unpack layer configuration --> general.
        self.dropout_rate: float = 0.0
        self.num_heads: int = 1
        self.k_nbrs: int = 50
        self._unpack_configuration(config=layer_config.get('general'))

        # unpack layer configuration --> parameters.
        self.units: int | None = None
        self.activation: str | None = None
        self.kernel_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        # assert units and number of heads aligns
        assert self.units % self.num_heads == 0, "Fusion units must be divisible by the number of heads.."
        self.head_units = self.units // self.num_heads
        self.units_ffn = 4 * self.units

        # initiate variables to build - fused projection
        self.fused_proj = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, name=f'{self.name}_fused_projection'
        )

        # initiate variables to build - self attention.
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
            bias_initializer=self.bias_initializer, use_bias=False,  name=f'{self.name}_output_projection'
        )

        # initiate variables to build - dropout and layer norm.
        self.layer_norm = tf.keras.layers.LayerNormalization(name=f'{self.name}_layer_normalisation')
        self.attn_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_attn_dropout')

        # initiate variables to build - feed-forward network.
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
        """
        Build TargetAttentionLayer.
        :param input_shape: tuple of input shapes:
            - fused_features: gnn and rnn fused features [batch, num_trades, d_f]
            - adjacency: adjacency matrix [num_trades, num_trades]
            - target indices: indices of target trades in adjacency
        :return:
        """
        # extract input shapes.
        _, _, _ = input_shape

        # update layer build flag.
        super().build(input_shape)

    def call(
            self, inputs: Tuple[tf.Tensor, Union[tf.Tensor, tf.SparseTensor], tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass for target attention layer.
        :param inputs: tuple of inputs:
            - fused_features: gnn and rnn fused features [batch, num_trades, d_f]
            - adjacency: adjacency matrix [num_trades, num_trades]
            - target indices: indices of target trades in adjacency
        :param training: whether in training mode.
        :return:
        """
        # extract inputs
        fused_features, adjacency, target_idx = inputs

        # Slice fused features to target trades only.
        fused_features = tf.gather(fused_features, target_idx, axis=1)

        # Extract small [n_tgt, n_tgt] binary adjacency submatrix without
        # materializing the full [T, T] dense matrix.  We filter the sparse
        # indices to keep only entries where both row AND column are targets,
        # then remap to local [0..n_tgt-1] coordinates.
        adjacency = self._extract_target_submatrix(adjacency, target_idx)

        # extract dimensions.
        _, num_trades = tf.shape(fused_features)[0], tf.shape(fused_features)[1]

        # 0. project fused features to attention space.
        fused = self.fused_proj(fused_features)

        # 1. project to queries, key, values.
        query = self.q_dense(fused)         # [b, t, d_attn]
        key = self.k_dense(fused)           # [b, t, d_attn]
        value = self.v_dense(fused)         # [b, t, d_attn]

        # 2. split heads
        query_h, key_h, value_h = (self._split_heads(x) for x in (query, key, value))

        # 3. apply core attention layer calculation. (calculates scores, adj mask, weights and context).
        context = self._core_calc(
            q=query_h, k=key_h, v=value_h, adjacency=adjacency, num_trades=num_trades, training=training
        )

        # 4. combine heads
        attn_out = self._combine_heads(context)         # [b, t, d_attn]
        attn_out = self.out_dense(attn_out)             # [b, t, d_attn]
        attn_out = self.layer_norm(fused + attn_out)    # residual + norm

        # 5. feed-forward sublayer + norm
        ffn = self.ffn_dense_1(attn_out)                # [b, t, d_attn]
        ffn = self.ffn_dropout(ffn, training=training)
        ffn = self.ffn_dense_2(ffn)                     # [b, t, d_attn]
        return self.ffn_norm(attn_out + ffn)            # [b, t, d_attn]

    def compute_output_shape(
            self, input_shape: Tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]
    ) -> tf.TensorShape:
        """
        Compute output shape of the layer.

        :param input_shape:
        :return:
        """
        # extract input shapes.
        _, _, _ = input_shape
        return tf.TensorShape([None, None, self.units])

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for serializing the layer.

        :return:
        """
        config = super(TargetAttentionLayer, self).get_config()
        config.update({
            'layer_config': self.layer_config
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TargetAttentionLayer":
        """
        Instantiates the TargetAttentionLayer from its config.

        :param config:
        :return:
        """
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
            adjacency: Union[tf.Tensor, tf.SparseTensor], num_trades, training: bool = False
    ) -> tf.Tensor:
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
        :return: context tensor [B, h, n_tgt, d_h]
        """
        if isinstance(adjacency, tf.SparseTensor):
            return self._sparse_target_attention(q, k, v, adjacency, num_trades, training)
        return self._dense_target_attention(q, k, v, adjacency, num_trades, training)

    def _dense_target_attention(
            self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
            adjacency: tf.Tensor, num_trades, training: bool = False
    ) -> tf.Tensor:
        """Fallback O(n_tgt^2) attention for dense adjacency submatrices."""
        scores = tf.matmul(q, k, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.head_units, scores.dtype))

        mask = tf.cast(adjacency > 0, scores.dtype)
        mask = tf.reshape(mask, [1, 1, num_trades, num_trades])

        very_neg = tf.cast(-1e9, scores.dtype)
        scores_masked = tf.where(mask > 0, scores, very_neg)
        weights = tf.nn.softmax(scores_masked, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        context = tf.matmul(weights, v)
        return context

    def _sparse_target_attention(
            self, q: tf.Tensor, k_proj: tf.Tensor, v: tf.Tensor,
            adjacency: tf.SparseTensor, num_trades, training: bool = False
    ) -> tf.Tensor:
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
        return context

    def _combine_heads(self, x: tf.Tensor) -> tf.Tensor:
        """
        Inverse of _split_heads()
        :param x:
        :return:
        """
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        return tf.reshape(x, [b, t, self.units])

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """
        Split last fim into (num_heads, head_dim) and transpose to [b, h, t, d_h]
        :param x:
        :return:
        """
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.num_heads, self.head_units])
        return tf.transpose(x, perm=[0, 2, 1, 3])

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
