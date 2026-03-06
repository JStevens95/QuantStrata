"""
PnL projection layer: per-target baseline + residual MLP correction.

Train targets get a dedicated learned kernel/bias (optionally unit-norm + gain).
New targets inherit baselines via kNN-weighted blending of train outputs in
attribute space.  An optional attention-conditioned scale/bias can modulate
new-target predictions post-hoc.

PyTorch port of the TensorFlow TargetPnlOutput layer.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Mapping from activation name strings to callable torch functions.
_ACTIVATION_MAP = {
    "relu": F.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.2),
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "elu": F.elu,
    "selu": F.selu,
    "linear": lambda x: x,
    None: lambda x: x,
}


def _get_activation(name: Optional[str]):
    """Return the torch activation function for a given name string."""
    if name in _ACTIVATION_MAP:
        return _ACTIVATION_MAP[name]
    raise ValueError(f"Unsupported activation: {name}. Choose from {list(_ACTIVATION_MAP.keys())}")


class TargetPnlOutput(nn.Module):
    """
    Per-target PnL logits: baseline (trained kernel or kNN-interpolated) + residual MLP.

    Baseline path:
        Train targets  -> kernel @ attn_vector + bias  (or unit-norm kernel * gain + bias)
        New targets    -> kNN-weighted blend of train-target baselines in attribute space

    Residual path:
        MLP( attn_features || target_attributes ) -> per-target correction added to baseline

    Optional attention conditioning:
        New targets only: learned softplus-scale and/or additive bias from attention features.

    Config (``layer_config['general']``):
        - ``baseline_trade_count`` (optional): number of train targets with dedicated
          kernel/bias.  If omitted, inferred from ``attn_features.shape[1]`` on the
          first forward pass (matching the TF ``build()`` behaviour).
        - ``attn_dim`` (optional): dimensionality of the attention feature vectors.
          If omitted, inferred from ``attn_features.shape[-1]`` on the first forward
          pass.
    """

    def __init__(self, layer_config: Dict[str, Any], name: str = "target_pnl_output") -> None:
        super().__init__()

        self.layer_config: Dict[str, Any] = layer_config
        self.layer_name: str = name

        # ----- General hyperparameters (overwritten by config via _unpack) -----
        self.baseline_new_mode: str = "output_mix"
        self.baseline_trade_count: Optional[int] = None     # optional, inferred if omitted
        self.attn_dim: Optional[int] = None                # optional, inferred if omitted
        self.dropout_rate: float = 0.0
        self.use_baseline_weight_norm: bool = False
        self.use_attn_scale_new: bool = False
        self.use_attn_bias_new: bool = False
        self.residual_new_damp: float = 1.0
        # kNN settings for new-target baseline interpolation
        self.knn_k: int = 4
        self.knn_power: float = 2.0
        self.knn_temperature: float = 5.0
        self.knn_mode: str = "cosine_softmax"
        self.knn_eps: float = 1e-8
        self._unpack_configuration(config=layer_config.get("general"))

        # ----- Dense-layer parameters -----
        self.units: Optional[int] = None
        self.activation: Optional[str] = None
        self.kernel_initializer: Optional[str] = None
        self.bias_initializer: Optional[str] = None
        self._unpack_configuration(config=layer_config.get("parameters"))

        # Inverse softplus(1.0) — used to initialise gain so softplus(gain) ≈ 1.
        self.softplus_inv: float = float(np.log(np.expm1(1.0)))

        n0 = self.baseline_trade_count
        d = self.attn_dim

        # ----- Baseline parameters: per-train-target learned projection -----
        # Kernels need both n0 and d; biases/gain need only n0.
        # When a dimension is unknown, use UninitializedParameter (materialized
        # on first forward, in-place, so the optimizer keeps tracking it).
        if n0 is not None and d is not None:
            self._baseline_kernels = nn.Parameter(torch.empty(n0, d))
            nn.init.xavier_uniform_(self._baseline_kernels)
        else:
            self._baseline_kernels = nn.UninitializedParameter()

        if n0 is not None:
            self._baseline_biases = nn.Parameter(torch.zeros(n0))
        else:
            self._baseline_biases = nn.UninitializedParameter()

        # Optional weight-norm gain: softplus(gain) * unit-norm(kernel) decomposition.
        self._baseline_gain: Optional[nn.Parameter] = None
        if self.use_baseline_weight_norm:
            if n0 is not None:
                self._baseline_gain = nn.Parameter(
                    torch.full((n0,), self.softplus_inv)
                )
            else:
                self._baseline_gain = nn.UninitializedParameter()

        # Resolve the activation function once for the residual MLP.
        self._act_fn = _get_activation(self.activation)

        # ----- Residual MLP: fc1 -> activation -> dropout -> fc2(linear, 1 output) -----
        self._residual_fc_1 = nn.LazyLinear(self.units)
        self._residual_dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else None
        self._residual_fc_2 = nn.LazyLinear(1)

        # ----- Optional attention-conditioned scale / bias for new targets -----
        self._attn_scale_dense: Optional[nn.Module] = None
        if self.use_attn_scale_new:
            if d is not None:
                self._attn_scale_dense = nn.Linear(d, 1)
            else:
                self._attn_scale_dense = nn.LazyLinear(1)

        self._attn_bias_dense: Optional[nn.Module] = None
        if self.use_attn_bias_new:
            if d is not None:
                self._attn_bias_dense = nn.Linear(d, 1)
            else:
                self._attn_bias_dense = nn.LazyLinear(1)

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _materialize_lazy_params(
        self, num_targets: int, attn_dim: int
    ) -> None:
        """Resolve unknown dimensions and materialize deferred parameters.

        Mirrors TF ``build()`` logic: prefer config values, fall back to input
        shapes.  Called once on the first forward pass when either
        ``baseline_trade_count`` or ``attn_dim`` was omitted from config.

        Persists resolved values back into ``layer_config`` so that
        ``get_config()`` round-trips correctly (same as the TF version).
        """
        if self.baseline_trade_count is None:
            self.baseline_trade_count = num_targets
        if self.attn_dim is None:
            self.attn_dim = attn_dim

        # Persist to layer_config for serialization fidelity.
        general = self.layer_config.get("general")
        if isinstance(general, dict):
            general.setdefault("baseline_trade_count", self.baseline_trade_count)
            general.setdefault("attn_dim", self.attn_dim)

        n0 = self.baseline_trade_count
        d = self.attn_dim

        if isinstance(self._baseline_kernels, nn.UninitializedParameter):
            self._baseline_kernels.materialize((n0, d))
            nn.init.xavier_uniform_(self._baseline_kernels)
        if isinstance(self._baseline_biases, nn.UninitializedParameter):
            self._baseline_biases.materialize((n0,))
            nn.init.zeros_(self._baseline_biases)
        if self._baseline_gain is not None and isinstance(
            self._baseline_gain, nn.UninitializedParameter
        ):
            self._baseline_gain.materialize((n0,))
            nn.init.constant_(self._baseline_gain, self.softplus_inv)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute per-target PnL logits: baseline + residual.

        :param inputs: Tuple of three tensors:
            - trade_features [N, p]:           static attributes for all N trades.
            - attn_features  [B, n_targets, d]: attention-derived features per target.
            - tgt_idx        [n_targets]:       indices into trade_features for targets.
        :returns: PnL predictions [B, n_targets].
        """
        trade_features, attn_features, tgt_idx = inputs

        if self.baseline_trade_count is None or self.attn_dim is None:
            self._materialize_lazy_params(
                num_targets=attn_features.shape[1],
                attn_dim=attn_features.shape[-1],
            )

        batch = attn_features.shape[0]          # B — batch size
        num_targets = attn_features.shape[1]    # n — number of target trades

        # Look up static attribute vectors for each target trade.
        target_attrs = trade_features[tgt_idx]  # [n_targets, p]

        # ---- Residual MLP: context-dependent correction for every target ----
        # Broadcast target attrs across batch: [n_targets, p] -> [B, n_targets, p].
        attrs_batched = target_attrs.unsqueeze(0).expand(batch, -1, -1)
        # Concatenate attention features and attributes as MLP input.
        resid_input = torch.cat([attn_features, attrs_batched], dim=-1)  # [B, n, d+p]
        # Two-layer MLP: Linear -> activation -> dropout -> Linear(1) -> squeeze.
        residual_fc = self._act_fn(self._residual_fc_1(resid_input))     # [B, n, units]
        if self._residual_dropout is not None:
            residual_fc = self._residual_dropout(residual_fc)
        residual = self._residual_fc_2(residual_fc).squeeze(-1)          # [B, n]

        # ---- Baseline for the first n0 (train) targets ----
        # n0 = number of targets that have dedicated kernels.
        n0 = min(self.baseline_trade_count, num_targets)
        attn_train = attn_features[:, :n0, :]  # [B, n0, d]

        if self.use_baseline_weight_norm:
            # Unit-norm kernel * softplus(gain) decomposition.
            base_train = self._baseline_train_gain(attn_train, n0)
        else:
            # Standard dot-product baseline: kernel @ attn + bias per train target.
            base_train = (
                torch.einsum("bna,na->bn", attn_train, self._baseline_kernels[:n0, :])
                + self._baseline_biases[None, :n0]
            )  # [B, n0]

        # ---- Split residual and attributes into train / new slices ----
        n_new = num_targets - n0
        resid_train = residual[:, :n0]
        resid_new = residual[:, n0:]
        attrs_train = target_attrs[:n0, :]
        attrs_new = target_attrs[n0:, :]

        # ---- New-target baseline: kNN-weighted blend of train baselines ----
        if self.baseline_new_mode.lower() == "output_mix":
            if n_new > 0:
                base_new, resid_new = self._baseline_new_output_mix(
                    base_train, resid_new, attrs_train, attrs_new
                )
            else:
                # All targets are train targets — nothing to interpolate.
                base_new, resid_new = self._baseline_new_empty(
                    batch, n_new, resid_new, attn_features.dtype
                )
        else:
            raise ValueError(f"Unknown baseline mode: '{self.baseline_new_mode}'")

        # ---- Combine baseline + residual -> logits ----
        baseline = torch.cat([base_train, base_new], dim=1)   # [B, n]
        residual = torch.cat([resid_train, resid_new], dim=1) # [B, n]
        logits = baseline + residual                           # [B, n]

        # Optional post-hoc attention-conditioned scale/bias for new targets.
        preds = self._apply_attention_conditioning(logits, attn_features, n0)
        return preds

    # ------------------------------------------------------------------
    # Baseline helpers
    # ------------------------------------------------------------------

    def _baseline_train_gain(self, attn_train: torch.Tensor, n0: int) -> torch.Tensor:
        """
        Weight-norm baseline: unit-norm(kernel) * softplus(gain) * dot(attn) + bias.

        Decomposes each kernel into direction (unit norm) and amplitude (positive gain)
        so that gradient updates to direction and magnitude are decoupled.

        :param attn_train: Attention features for train targets [B, n0, d].
        :param n0: Number of train targets to use.
        :returns: Baseline predictions [B, n0].
        """
        k = self._baseline_kernels[:n0, :]                     # [n0, d]
        k_norm = torch.norm(k, dim=1, keepdim=True)            # [n0, 1]
        k_unit = k / (k_norm + self.knn_eps)                   # [n0, d] unit-norm

        # Compute per-target gain: softplus ensures strictly positive amplitude.
        if self.use_baseline_weight_norm and (self._baseline_gain is not None):
            g = F.softplus(self._baseline_gain[:n0])            # [n0]
        else:
            g = torch.ones(n0, dtype=attn_train.dtype, device=attn_train.device)

        # Dot product of each target's attention vector with its unit-norm kernel,
        # scaled by gain, plus per-target bias.
        base_train = (
            g[None, :] * torch.sum(attn_train * k_unit[None, :, :], dim=-1)
            + self._baseline_biases[None, :n0]
        )  # [B, n0]
        return base_train

    def _baseline_new_empty(
        self,
        batch_size: int,
        n_new: int,
        residual_new: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fallback when there are no new targets: zero baseline, damped residual.

        :returns: (base_new [B, 0], resid_new [B, 0]) — empty along target dim.
        """
        base_new = torch.zeros(batch_size, n_new, dtype=dtype, device=residual_new.device)
        resid_new = residual_new * self.residual_new_damp
        return base_new, resid_new

    def _baseline_new_output_mix(
        self,
        base_train: torch.Tensor,
        residual_new: torch.Tensor,
        attrs_train: torch.Tensor,
        attrs_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute new-target baselines as a kNN-weighted mix of train baselines.

        For each new target, find the k nearest train targets in attribute space
        and blend their baseline predictions using the kNN weights.

        :param base_train:    Train-target baselines [B, n0].
        :param residual_new:  Residual MLP output for new targets [B, n_new].
        :param attrs_train:   Attribute vectors of train targets [n0, p].
        :param attrs_new:     Attribute vectors of new targets [n_new, p].
        :returns: (base_new [B, n_new], resid_new [B, n_new]).
        """
        # Get kNN indices and weights: idx [n_new, k], w [n_new, k].
        idx, w = self._knn_weights(attrs_new, attrs_train)

        # Gather train baselines for the k nearest neighbors of each new target.
        base_knn = base_train[:, idx]                            # [B, n_new, k]
        # Weighted sum: blend neighbor baselines using kNN weights.
        base_new = torch.sum(base_knn * w[None, :, :], dim=-1)  # [B, n_new]
        resid_new = residual_new * self.residual_new_damp
        return base_new, resid_new

    # ------------------------------------------------------------------
    # Attention conditioning (post-hoc modulation for new targets)
    # ------------------------------------------------------------------

    def _apply_attention_conditioning(
        self,
        logits_bn: torch.Tensor,
        attention_feats: torch.Tensor,
        n_train_targets: int,
    ) -> torch.Tensor:
        """
        Optional post-hoc modulation for new targets only.

        Train targets (indices < n_train_targets) pass through unchanged (scale=1, bias=0).
        New targets (indices >= n_train_targets) receive learned attention-conditioned
        scale and/or bias:
            out_new = softplus(w_s @ attn) * logit + w_b @ attn

        :param logits_bn:       Combined baseline + residual logits [B, n].
        :param attention_feats:  Attention feature matrix [B, n, d].
        :param n_train_targets:  Count of train targets (left partition of target axis).
        :returns: Modulated predictions [B, n].
        """
        if not (self.use_attn_scale_new or self.use_attn_bias_new):
            return logits_bn

        batch = attention_feats.shape[0]
        num_trades = attention_feats.shape[1]
        out = logits_bn

        # Binary masks: left = train targets (identity), right = new targets (modulated).
        cols = torch.arange(num_trades, device=out.device)[None, :]       # [1, n]
        left_mask = (cols < n_train_targets).to(out.dtype)                # [1, n]
        right_mask = 1.0 - left_mask                                      # [1, n]
        left_mask = left_mask.expand(batch, -1)                           # [B, n]
        right_mask = right_mask.expand(batch, -1)                         # [B, n]

        # Scale path: softplus ensures strictly positive scale.
        if self.use_attn_scale_new:
            raw_scale = self._attn_scale_dense(attention_feats)           # [B, n, 1]
            scale = F.softplus(raw_scale.squeeze(-1))                     # [B, n]
            # Train targets keep scale=1; new targets get learned scale.
            scale_full = left_mask * 1.0 + right_mask * scale
            out = out * scale_full

        # Bias path: train targets get zero bias; new targets get learned bias.
        if self.use_attn_bias_new:
            raw_bias = self._attn_bias_dense(attention_feats)             # [B, n, 1]
            bias = raw_bias.squeeze(-1)                                   # [B, n]
            bias_full = right_mask * bias
            out = out + bias_full

        return out

    # ------------------------------------------------------------------
    # kNN helpers
    # ------------------------------------------------------------------

    def _knn_weights(
        self, attrs_new: torch.Tensor, attrs_train: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute kNN indices and weights from new targets to train targets.

        Modes:
            cosine_softmax — cosine similarity with temperature-scaled softmax weights.
            idw            — inverse-distance weighting in Euclidean space.

        :param attrs_new:   Attribute vectors of new targets [n_new, p].
        :param attrs_train: Attribute vectors of train targets [n0, p].
        :returns: (idx [n_new, k], weights [n_new, k]).
        """
        if self.knn_mode.lower() == "cosine_softmax":
            # L2-normalise both sets so dot product = cosine similarity.
            x_n = F.normalize(attrs_new, p=2, dim=1)             # [n_new, p]
            x_t = F.normalize(attrs_train, p=2, dim=1)           # [n0, p]
            sims = x_n @ x_t.T                                   # [n_new, n0]
            # Pick top-k most similar train targets per new target.
            k = min(self.knn_k, sims.shape[1])
            vals, idx = torch.topk(sims, k=k, dim=1, sorted=False)
            # Temperature-scaled softmax converts similarities to weights.
            weights = torch.softmax(vals * self.knn_temperature, dim=1)
            return idx, weights

        elif self.knn_mode.lower() == "idw":
            # Euclidean distance matrix [n_new, n0].
            d = self._pairwise_distances(attrs_new, attrs_train)
            # Pick k nearest (smallest distance = largest negative distance).
            k = min(self.knn_k, d.shape[1])
            neg_d = -d
            vals, idx = torch.topk(neg_d, k=k, dim=1, sorted=False)
            # Inverse-distance weighting: w_j = 1/d_j^power, normalised.
            d_k = torch.clamp(-vals, min=self.knn_eps)
            raw_w = 1.0 / torch.pow(d_k, self.knn_power)
            weights = raw_w / (torch.sum(raw_w, dim=1, keepdim=True) + 1e-12)
            return idx, weights

        else:
            raise ValueError(
                f"Unknown knn_mode '{self.knn_mode}'. Use 'cosine_softmax' or 'idw'."
            )

    def _pairwise_distances(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Pairwise Euclidean distance: sqrt(||x_i||^2 + ||y_j||^2 - 2·x_i·y_j).

        :param x: Query points [m, p].
        :param y: Reference points [n, p].
        :returns: Distance matrix [m, n].
        """
        x2 = torch.sum(x ** 2, dim=1, keepdim=True)   # [m, 1]
        y2 = torch.sum(y ** 2, dim=1, keepdim=True)    # [n, 1]
        cross = x @ y.T                                 # [m, n]
        d2 = torch.clamp(x2 + y2.T - 2.0 * cross, min=0.0)
        return torch.sqrt(d2 + self.knn_eps)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration dictionary for serialization."""
        return {"layer_config": self.layer_config, "name": self.layer_name}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TargetPnlOutput":
        """Reconstruct a TargetPnlOutput from a configuration dictionary."""
        return cls(**config)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from a config sub-dict (key=value -> self.key = value)."""
        for k, v in config.items():
            setattr(self, k, v)
