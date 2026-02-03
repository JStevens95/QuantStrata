import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple

# define logging at module level.
logger = logging.getLogger(__name__)


class TargetPnlOutput(tf.keras.layers.Layer):
    """
    Dynamic per target linear pnl projection layer.

    Replicates target-specific kernels and biases without fixing number of targets at calibration.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs):
        """
        Initialise TargetPnlOutput layer.

        :param layer_config: dictionary containing general layer configuration & parameters.
        """
        # init call to super class
        super().__init__(**kwargs)

        # initiate required variables.
        self.layer_config: Dict[str, Any] = layer_config
        self.kwargs = kwargs

        # unpack layer configuration --> general.
        self.baseline_new_mode: str | None = 'output_mix'
        self.baseline_trade_count: int | None = None
        self.dropout_rate: float = 0.0
        self.new_target_mode: str | None = None
        self.use_baseline_norm: bool | None = None
        self.use_attn_scale: bool | None = None
        self.use_attn_bias: bool | None = None
        self.residual_new_damp: float = 1.0
        # kNN settings
        self.knn_k: int | None = 4
        self.knn_power: float | None = 2.0
        self.knn_temperature: float | None = 5.0
        self.knn_mode: str | None = "cosine_softmax"
        self.knn_eps: float = 1e-8
        self._unpack_configuration(config=layer_config.get('general'))

        # unpack layer configuration --> parameters.
        self.units: int | None = None
        self.activation: str | None = None
        self.kernel_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get('parameters'))

        # initiate variables to build.
        self.softplus_inv = float(np.log(np.expm1(1.0)))
        self._residual_fc_1: tf.keras.layers.Dense | None = None
        self._residual_fc_2: tf.keras.layers.Dense | None = None
        self._baseline_kernels: tf.Variable | None = None
        self._baseline_biases: tf.Variable | None = None
        self._attn_scale_dense: tf.keras.layers.Dense | None = None
        self._attn_bias_dense: tf.keras.layers.Dense | None = None
        self._baseline_gain: tf.Variable | None = None


    def build(self, input_shape: Tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]) -> None:
        """
        Build the layer with the given input shape.

        :param input_shape: tuple of input shapes:
             - trade features: [num_trades, features]
             - attention features: [batch, num_targets, attn_units]
             - target indices: [num_targets]
        :return:
        """
        # extract input shapes.
        trade_features, attn_features, tgt_idx = input_shape

        # define dimensions.
        attn_dim = attn_features[-1]

        # baseline_trade_count resolution
        cfg_n0 = self.baseline_trade_count  # possibly provided via config
        if cfg_n0 is None:
            if attn_features[1] is None:
                raise ValueError(
                    "TargetPnlOutput: 'baseline_trade_count' must be provided in config "
                    "when num_targets is dynamic at build time."
                )
            self.baseline_trade_count = int(attn_features[1])
        else:
            self.baseline_trade_count = int(cfg_n0)

        # residual mlp: (attention || attrs) --> hidden --> scale per target.
        self._residual_fc_1 = tf.keras.layers.Dense(
            units=self.units, activation=self.activation, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, name=f'{self.name}_residual_fc_1'
        )
        self._residual_fc_2 = tf.keras.layers.Dense(
            units=1, activation=None, name=f'{self.name}_residual_fc_2'
        )

        # trainable baseline weights for trained targets only.
        self._baseline_kernels = self.add_weight(
            shape=(self.baseline_trade_count, attn_dim), initializer='glorot_uniform', trainable=True,
            name=f'{self.name}_baseline_kernels'
        )
        self._baseline_biases = self.add_weight(
            shape=(self.baseline_trade_count, ), initializer='zeros', trainable=True,
            name=f'{self.name}_baseline_biases'
        )

        # learnable positive per-train-target gain (amplitude) initialized ~1
        if self.use_baseline_norm:
            self._baseline_gain = self.add_weight(
                shape=(self.baseline_trade_count, ), initializer=tf.keras.initializers.Constant(self.softplus_inv),
                trainable=True, name=f'{self.name}_baseline_gain'
            )

        # new target only attention conditioned output scale / bias heads (optional)
        if self.use_attn_scale:
            self._attn_scale_dense = tf.keras.layers.Dense(
                units=1, activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                name=f'{self.name}_attn_scale_dense'
            )
        if self.use_attn_bias:
            self._attn_bias_dense = tf.keras.layers.Dense(
                units=1, activation=None, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                name=f'{self.name}_attn_bias_dense'
            )

        # update layer build flag.
        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass of TargetPnlOutput layer.

        :param inputs:
        :param training:
        :return:
        """
        # extract inputs.
        trade_features, attn_features, tgt_idx = inputs

        # extract variables from dimensions.
        batch = tf.shape(attn_features)[0]
        num_targets = tf.shape(attn_features)[1]

        # filter trade features for only target trades in current order
        target_attrs = tf.gather(trade_features, tgt_idx, axis=0)

        # ---- residuals (all targets). ---- #
        attrs_batched = tf.tile(tf.expand_dims(target_attrs, axis=0), [batch, 1, 1])    # [batch, n_targets, p]
        resid_input = tf.concat([attn_features, attrs_batched], axis=-1)
        residual_fc = self._residual_fc_1(resid_input)                                        # [batch, n_targets, h]
        residual = tf.squeeze(self._residual_fc_2(residual_fc), axis=-1)                      # [batch, n_targets]

        # ---- baseline for training targets. ---- #
        n0_cfg = self.baseline_trade_count
        n0 = num_targets if n0_cfg is None else tf.minimum(tf.cast(n0_cfg, tf.int32), num_targets)
        attn_train = attn_features[:, :n0, :]                                       # [batch, n0, a]
        if self.use_baseline_norm:
            base_train = self._baseline_train_gain(attn_train, n0)
        else:
            base_train = (tf.einsum("bna, na->bn", attn_train, self._baseline_kernels[:n0, :])
                          + self._baseline_biases[None, :n0])

        # ---- new target slices ---- #
        n_new = num_targets - n0
        attn_new = attn_features[:, n0:, :]             # [batch, n_new, a]
        resid_train = residual[:, :n0]                  # [batch, n0]
        resid_new = residual[:, n0:]                    # [batch, n_new]
        attrs_train = target_attrs[:n0, :]              # [n0, p]
        attrs_new = target_attrs[n0:, :]                # [n_new, p]

        # ---- route new-target strategy ---- #
        def do_none():
            return self._baseline_new_empty(batch, n_new, resid_new, attn_features.dtype)
        def do_output_mix():
            return self._baseline_new_output_mix(base_train, resid_new, attrs_train, attrs_new)

        # choose strategy via baseline model.
        if self.baseline_new_mode.lower() == 'output_mix':
            base_new, resid_new = tf.cond(n_new > 0, true_fn=do_output_mix, false_fn=do_none)
        else:
            raise ValueError(f"Unknown baseline mode: '{self.baseline_new_mode}'")

        # combine train and new parts (baseline & residual), then sum --> logits in z-space.
        baseline = tf.concat([base_train, base_new], axis=1)        # [batch, n]
        residual = tf.concat([resid_train, resid_new], axis=1)      # [batch, n]
        logits = baseline + residual

        # optional attention-conditioned new-only post modulation (scale/bias)
        preds = self._apply_attention_conditioning(logits, attn_features, n0)
        return preds

    @staticmethod
    def compute_output_shape(input_shape: Tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]) -> tf.TensorShape:
        """
        Compute output shape of the layer.

        :param input_shape:
        :return:
        """
        # extract input shapes.
        trade_feat, attn_feat, _ = input_shape
        return tf.TensorShape([attn_feat[0], attn_feat[1]])

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for serializing the layer.

        :return:
        """
        config = super(TargetPnlOutput, self).get_config()
        config.update({
            'layer_config': self.layer_config
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TargetPnlOutput":
        """
        Instantiates the TargetPnlOutput from its config.

        :param config:
        :return:
        """
        return cls(**config)

    def _apply_attention_conditioning(
            self, logits_bn: tf.Tensor, attention_feats: tf.Tensor, n_train_targets: tf.Tensor
    ) -> tf.Tensor:
        """
        Optionally apply attention-conditioned outer scaler / bias to new targets only.

        Train targets are left untouched.

        :param logits_bn:
        :param attention_feats:
        :param n_train_targets:
        :return:
        """
        if not (self.use_attn_scale or self.use_attn_bias):
            return logits_bn

        # extract variables from input shapes.
        batch = tf.shape(attention_feats)[0]
        num_trades = tf.shape(attention_feats)[1]

        # initialise output as input.
        out = logits_bn

        # build masks by constructing left/right partitions once.
        cols= tf.range(num_trades)[None, :]
        left_mask = tf.cast(cols < n_train_targets, out.dtype)
        right_mask = 1.0 - left_mask
        # tile to [b, n] for clarity.
        left_mask = tf.tile(left_mask, [batch, 1])
        right_mask = tf.tile(right_mask, [batch, 1])

        # scale path (new only)
        if self.use_attn_scale:
            # raw scale from attention features - positive via softplus
            raw_scale = self._attn_scale_dense(attention_feats)
            scale = tf.nn.softplus(tf.squeeze(raw_scale, axis=-1))
            scale_full = left_mask * 1.0 + right_mask * scale
            out = out * scale_full

        # bias path (new only)
        if self.use_attn_bias:
            # raw bias from attention features.
            raw_bias = self._attn_bias_dense(attention_feats)
            bias = tf.squeeze(raw_bias, axis=-1)

            # train slice get zero bias; new slice gets learned bias.
            bias_full = right_mask * bias
            out = out + bias_full
        return out

    def _baseline_new_empty(
            self, batch_size: tf.Tensor, n_new: tf.Tensor, residual_new: tf.Tensor, dtypes: tf.dtypes.DType
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        No baseline for new targets (zeros).

        Residual is optionally damped.

        :param batch_size:
        :param n_new:
        :param residual_new:
        :param dtypes:
        :return:
        """
        base_new = tf.zeros([batch_size, n_new], dtype=dtypes)
        resid_new = residual_new * self.residual_new_damp
        return base_new, resid_new

    def _baseline_new_output_mix(
            self, base_train: tf.Tensor, residual_new: tf.Tensor, attrs_train: tf.Tensor, attrs_new: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        New baseline via output-space mixing.

        Construct each new target's baseline as a convex combination of trained baseline using kNN weights in attribute
        space. Magnitude inherits the already learned, unit constant scale of trained outputs on the same scenarios.

        :param base_train:
        :param residual_new:
        :param attrs_train:
        :param attrs_new:
        :return:
        """
        # calculate kNN weights
        idx, w = self._knn_weights(attrs_new, attrs_train)          # [n_new, k], [n_new, k]

        # gather train baselines for each new target's neighbours
        base_knn = tf.gather(base_train, idx, axis=1)               # [b, n_new, k]

        # weighted sum over neighbours.
        base_new = tf.reduce_sum(base_knn * w[None, :, :], axis=-1)

        # optionally damp residual for new targets.
        resid_new = residual_new * self.residual_new_damp
        return base_new, resid_new

    def _baseline_train_gain(self, attn_train: tf.Tensor, n0: tf.Tensor):
        """
        Train baseline with kernel direction only + learned positive per-target gain.

        This makes amplitude explicit and stable; direction (shape) is from unit-norm kernels.

        :param attn_train: attention embeddings of train targets.
        :param n0: number of train targets in first slice of N
        :return:
        """
        # unit normalise each train kernel row --> carry direction not energy.
        k = self._baseline_kernels[:n0, :]
        k_norm = tf.norm(k, axis=1, keepdims=True)
        k_unit = k / (k_norm + self.knn_eps)

        # positive gain per training target.
        if self.use_baseline_norm and (self._baseline_gain is not None):
            g = tf.nn.softplus(self._baseline_gain[:n0])
        else:
            g = tf.ones(shape=(n0, ), dtype=attn_train.dtype)

        # project * scale + bias
        base_train = (g[None, :] * tf.reduce_sum(attn_train * k_unit[None, :, :], axis=-1) +
                      self._baseline_biases[None, :n0])
        return base_train

    def _knn_weights(self, attrs_new: tf.Tensor, attrs_train: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute neighbour indices and row-normalised weights from NEW to TRAIN targets in attribute space.

        :param attrs_new: attributes of NEW targets [n_new, p]
        :param attrs_train: attributes of TRAIN targets [n0, p]
        :return:
        """
        if self.knn_mode.lower() == 'cosine_softmax':
            # cosine similarities (unit-normalised embeddings).
            x_n = self._l2_normalise(attrs_new, axis=1)             # [n_new, p]
            x_t = self._l2_normalise(attrs_train, axis=1)           # [n0, p]
            sims = tf.matmul(x_n, x_t, transpose_b=True)

            # select top-k neighouts by similarity.
            k = tf.minimum(self.knn_k, tf.shape(sims)[1])
            vals, idx = tf.math.top_k(sims, k=k, sorted=False)      # vals; [n_new, k], idx; [n_new, p]

            # softmax ove similarities (temperature controls sharpness)
            weights = tf.nn.softmax(vals * self.knn_temperature, axis=1)    # [n_new, k]
            return idx, weights
        elif self.knn_mode.lower() == 'idw':
            # inverse distance weighting for euclidean distance.
            d = self._pairwise_distances(attrs_new, attrs_train)
            k = tf.minimum(self.knn_k, tf.shape(d)[1])

            # take top-k smallest distances via top_k on negatives.
            neg_d = -d
            vals, idx = tf.math.top_k(neg_d, k=k, sorted=False)
            d_k = tf.maximum(-vals, self.knn_eps)

            # calculate idw weights.
            raw_w = 1.0 / tf.pow(d_k, self.knn_power)
            weights = raw_w / (tf.reduce_sum(raw_w, axis=1, keepdims=True) + 1e-12)
            return idx, weights
        else:
            raise ValueError(f"Unknown knn_mode '{self.knn_mode}'. Use 'cosine_softmax'")

    @staticmethod
    def _l2_normalise(x: tf.Tensor, axis: int) -> tf.Tensor:
        return tf.math.l2_normalize(x, axis=axis)

    def _pairwise_distances(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Euclidean distance via ||x||^2 + ||y||^2 2x.y
        :param x:
        :param y:
        :return:
        """
        x2 = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        y2 = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
        cross = tf.matmul(x, y, transpose_b=True)
        d2 = tf.maximum(x2 + tf.transpose(y2) - 2.0 * cross, 0.0)
        return tf.sqrt(d2 + self.knn_eps)

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
