import logging
import tensorflow as tf
from typing import Dict, Any, Tuple, Union

# define logging at module level
logger = logging.getLogger(__name__)


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

        # filter fused features to look at target fused features only.
        if isinstance(adjacency, tf.SparseTensor):
            adjacency = tf.sparse.to_dense(adjacency)
        fused_features = tf.gather(fused_features, target_idx, axis=1)
        adjacency = tf.gather(tf.gather(adjacency, target_idx, axis=0), target_idx, axis=1)

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

    def _core_calc(
            self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, adjacency: tf.Tensor, num_trades, training: bool = False
    ) -> tf.Tensor:
        """
        Core fusion calculation.
        :param q:
        :param k:
        :param v:
        :param adjacency:
        :param num_trades:
        :param training:
        :return:
        """
        # 1) scaled dot-product
        scores = tf.matmul(q, k, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.head_units, tf.float32))

        # 2) build mask from adjacency
        adj_dense = tf.sparse.to_dense(adjacency) if isinstance(adjacency, tf.SparseTensor) else adjacency
        mask = tf.reshape(adj_dense, [1, 1, num_trades, num_trades])  # [1,1,T,T]

        # --- masked softmax ---
        very_neg = tf.cast(-1e9, scores.dtype)  # large negative sentinel
        scores_masked = tf.where(mask > 0, scores, very_neg)  # keep scores only where allowed

        # Keep masked entries strictly zero and handle zero-degree rows safely
        weights = tf.nn.softmax(scores_masked, axis=-1)
        weights = weights * tf.cast(mask, weights.dtype)
        weights = weights / (tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-9)

        # optional dropout
        weights = self.attn_dropout(weights, training=training)

        # 3) context
        context = tf.matmul(weights, v)
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
