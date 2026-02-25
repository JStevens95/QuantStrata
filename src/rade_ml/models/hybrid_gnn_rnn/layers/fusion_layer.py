import logging
import tensorflow as tf
from typing import Dict, Any, Tuple, Union

try:
    from keras.saving import register_keras_serializable
except ImportError:
    register_keras_serializable = tf.keras.saving.register_keras_serializable

_REGISTER_PACKAGE = "Tranql.RadeMl"

# define logging at module level.
logger = logging.getLogger(__name__)


@register_keras_serializable(package=_REGISTER_PACKAGE)
class FusionLayer(tf.keras.layers.Layer):
    """
    Per-target fusion layer combining GNN and RNN streams using multi-head cross attention and gating mechanism.

    This gate controls the flow of information from the GNN to the RNN, preserving the knowledge in the
    pre-trained LSTM while enhancing it with trade relationship information.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        """
        Initialise the FusionLayer.

        :param layer_config: dictionary containing general layer configuration & parameters.
        """
        # init call to super class
        super().__init__(**kwargs)

        # initiate required variables.
        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        # unpack layer configuration --> general.
        self.dropout_rate: float = 0.0
        self.fusion_mode: str | None = None
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

        # initiate variables to build - gnn & rnn projection.
        self.rnn_proj = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, name=f'{self.name}_rnn_projection'
        )
        self.gnn_proj = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, name=f'{self.name}_gnn_projection'
        )

        # initiate variables to build - cross attention.
        self.q_dense_rnn = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=False,
            name=f'{self.name}_query_projection_rnn'
        )
        self.q_dense_gnn = tf.keras.layers.Dense(
            units=self.units, activation=None, kernel_initializer=self.kernel_initializer, use_bias=False,
            name=f'{self.name}_query_projection_gnn'
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

        # initiate variables to build - gating mechanism.
        if (self.fusion_mode or "").lower() == 'gate':
            self.gate_dense = tf.keras.layers.Dense(
                units=1, activation=None, kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer, name=f'{self.name}_gate_projection'
            )
            self.gate_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, name=f'{self.name}_gate_dropout')

    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]) -> None:
        """
        Build the fusion layer.

        :param input_shape: tuple of input shapes.
            - gnn embedding: embedding output from GnnBlock [num_trades, gnn_units]
            - rnn embedding: embedding output from RnnBlock [batch, rnn_units]
            adjacency matrix: [num_trades, num_trades]
        :return:
        """
        # extract input shapes.
        _, _, _ = input_shape

        # update layer build flag.
        super().build(input_shape)

    def call(
            self, inputs: Tuple[tf.Tensor, tf.Tensor, Union[tf.Tensor, tf.SparseTensor]], training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass of FusionLayer.

        :param inputs: tuple of inputs.
            - gnn embedding: embedding output from GnnBlock [num_trades, gnn_units]
            - rnn embedding: embedding output from RnnBlock [batch, rnn_units]
            - adjacency matrix: [num_trades, num_trades]
        :param training: whether in training mode.
        :return:
        """
        # extract inputs
        gnn_features, rnn_features, adjacency = inputs

        # extract dimensions from inputs.
        num_trades, gnn_dim = tf.shape(gnn_features)[0], tf.shape(gnn_features)[1]
        batch, rnn_dim = tf.shape(rnn_features)[0], tf.shape(rnn_features)[1]

        # 1. broadcast gnn features --> [batch, num_trades, gnn_dim]
        gnn_bcst = tf.broadcast_to(tf.expand_dims(gnn_features, axis=0), [batch, num_trades, gnn_dim])
        gnn_emb = self.gnn_proj(gnn_bcst)

        # 2, broadcast rnn features --> [batch, num_trades, rnn_dim]
        rnn_bcst = tf.broadcast_to(tf.expand_dims(rnn_features, axis=1), [batch, num_trades, rnn_dim])
        rnn_emb = self.rnn_proj(rnn_bcst)

        # 3. project to query, key and values.
        query = self.q_dense_rnn(rnn_emb) + self.q_dense_gnn(gnn_emb)     # [batch, num_trades, d_f]
        key = self.k_dense(gnn_emb)                                       # [batch, num_trades, d_f]
        value = self.v_dense(gnn_emb)                                     # [batch, num_trades, d_f]

        # 4. split heads, compute scores.
        query_h, key_h, value_h = (self._split_heads(x) for x in (query, key, value))

        # 5. apply core attention layer calculation (calculates scores, adj mask, weights & context).
        context = self._core_calc(
            q=query_h, k=key_h, v=value_h, adjacency=adjacency, num_trades=num_trades, training=training
        )

        # 6. combine individual heads.
        fusion = self._combine_heads(context)
        fusion = self.out_dense(fusion)

        # 7. apply gating / concat logic for mixing.
        if (self.fusion_mode or "").lower() == "gate":
            gate_logit = self.gate_dense(tf.concat([fusion, rnn_emb], axis=-1))
            gate = tf.sigmoid(gate_logit)
            if self.gate_dropout is not None:
                gate = self.gate_dropout(gate, training=training)
            output = gate * fusion + (1.0 - gate) * rnn_emb
        elif (self.fusion_mode or "").lower() == "add":
            output = fusion + rnn_emb
        else:
            raise ValueError(f"Fusion mode {self.fusion_mode} not recognised..")
        return self.layer_norm(output)

    def compute_output_shape(
            self, input_shape: Tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]
    ) -> tf.TensorShape:
        """
        Compute output shape of the layer.

        :param input_shape: tuple of input shapes:
            - trade features [num_trades, features]
            - adjacency matrix [num_trades, num_trades]
        :return:
        """
        gnn_features, rnn_features, _ = input_shape
        return tf.TensorShape([rnn_features[0], gnn_features[0], self.units])

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for serializing the layer.

        :return:
        """
        config = super(FusionLayer, self).get_config()
        config.update({
            'layer_config': self.layer_config
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FusionLayer":
        """
        Instantiates the FusionLayer from its config.

        :param config:
        :return:
        """
        return cls(**config)

    def _core_calc(
            self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
            adjacency: Union[tf.Tensor, tf.SparseTensor], num_trades, training: bool = False
    ) -> tf.Tensor:
        """
        Core fusion calculation — dispatches to O(T*k) sparse neighborhood
        attention when the adjacency is a SparseTensor, or O(T^2) dense
        attention otherwise.

        :param q: query tensor  [B, h, T, d_h]
        :param k: key tensor    [B, h, T, d_h]
        :param v: value tensor  [B, h, T, d_h]
        :param adjacency: adjacency matrix (sparse or dense) [T, T]
        :param num_trades: scalar — number of trades T
        :param training: whether in training mode
        :return: context tensor [B, h, T, d_h]
        """
        if isinstance(adjacency, tf.SparseTensor):
            return self._sparse_nbr_attention(q, k, v, adjacency, num_trades, training)
        return self._dense_attention(q, k, v, adjacency, num_trades, training)

    def _dense_attention(
            self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
            adjacency: tf.Tensor, num_trades, training: bool = False
    ) -> tf.Tensor:
        """Fallback O(T^2) attention for dense adjacency matrices."""
        scores = tf.matmul(q, k, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.head_units, tf.float32))

        mask = tf.cast(adjacency > 0, tf.float32)
        mask = tf.reshape(mask, [1, 1, num_trades, num_trades])

        very_neg = tf.cast(-1e9, scores.dtype)
        scores = tf.where(mask > 0, scores, very_neg)
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        return tf.matmul(weights, v)

    def _sparse_nbr_attention(
            self, q: tf.Tensor, k_proj: tf.Tensor, v: tf.Tensor,
            adjacency: tf.SparseTensor, num_trades, training: bool = False
    ) -> tf.Tensor:
        """
        O(T * k) sparse neighborhood attention using the adjacency structure.

        Instead of materializing a [T, T] score matrix, each trade only
        attends to its k neighbors from the k-NN adjacency graph.  Memory
        goes from O(B * h * T^2) to O(B * h * T * k).

        :param q: query  [B, h, T, d_h]
        :param k_proj: key    [B, h, T, d_h]
        :param v: value  [B, h, T, d_h]
        :param adjacency: sparse adjacency [T, T] (reordered row-major)
        :param num_trades: scalar T
        :param training: whether in training mode
        :return: context [B, h, T, d_h]
        """
        rows = adjacency.indices[:, 0]   # int64
        cols = adjacency.indices[:, 1]   # int64
        num_trades_i64 = tf.cast(num_trades, tf.int64)

        # Per-row neighbor counts — uniform (k) for a k-NN graph, but
        # handles variable-degree gracefully.
        row_counts = tf.math.unsorted_segment_sum(
            tf.ones_like(rows, dtype=tf.int32), tf.cast(rows, tf.int32), num_trades
        )
        k = tf.reduce_max(row_counts)

        # Build padded [T, k] neighbor index array from the sparse structure.
        # from_value_rowids requires sorted (non-decreasing) row ids — guaranteed
        # because the SparseTensor was reordered to row-major.
        nbr_ragged = tf.RaggedTensor.from_value_rowids(cols, rows, nrows=num_trades_i64)
        nbr_idx = tf.cast(nbr_ragged.to_tensor(default_value=tf.constant(0, cols.dtype)), tf.int32)
        nbr_mask = tf.sequence_mask(row_counts, maxlen=k, dtype=tf.float32)

        # Gather neighbor keys and values along the T axis.
        # tf.gather(params=[B,h,T,d_h], indices=[T,k], axis=2) -> [B,h,T,k,d_h]
        k_nbr = tf.gather(k_proj, nbr_idx, axis=2)
        v_nbr = tf.gather(v, nbr_idx, axis=2)

        # Scores: dot(query, neighbor_key) -> [B, h, T, k]
        q_exp = tf.expand_dims(q, axis=3)                          # [B, h, T, 1, d_h]
        scores = tf.reduce_sum(q_exp * k_nbr, axis=-1)             # [B, h, T, k]
        scores /= tf.math.sqrt(tf.cast(self.head_units, tf.float32))

        # Mask padded neighbor positions with -1e9 so softmax drives them to ~0.
        mask_bcast = tf.reshape(nbr_mask, [1, 1, num_trades, k])   # [1, 1, T, k]
        very_neg = tf.cast(-1e9, scores.dtype)
        scores = tf.where(mask_bcast > 0, scores, very_neg)

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, training=training)

        # Weighted context: [B, h, T, d_h]
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