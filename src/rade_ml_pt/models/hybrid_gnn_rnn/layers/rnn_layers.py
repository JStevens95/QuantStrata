"""
RNN block for the Hybrid GNN-RNN model: LSTM, BiLSTM, or GRU.

Compresses P&L time series [B, S, T_e] into a fixed-length temporal embedding [B, d_r].
PyTorch LSTM/GRU use hardcoded activations (tanh for cell, sigmoid for gates) inside
fused cuDNN kernels. The ``activation`` config is used by the dense fallback branch.
"""
import logging
import torch
import torch.nn as nn

from typing import Dict, Any, Optional, Type

logger = logging.getLogger(__name__)

_RECURRENT_TYPES = {"lstm", "bilstm", "gru"}

_ACTIVATION_MAP: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
}


def _resolve_act_module(name: Optional[str]) -> Optional[nn.Module]:
    """Return an instantiated activation ``nn.Module`` for *name*, or None."""
    act = (name or "").lower()
    if not act or act == "linear":
        return None
    cls = _ACTIVATION_MAP.get(act)
    if cls is None:
        raise ValueError(
            f"Unsupported activation '{name}'. Supported: {sorted(_ACTIVATION_MAP.keys())}"
        )
    return cls()


class RnnBlock(nn.Module):
    """
    Stack of L recurrent layers (LSTM, BiLSTM, or GRU) for temporal P&L compression.

    PyTorch LSTM/GRU supports stacking natively via the ``num_layers`` parameter, so
    no manual loop is needed. The final hidden state is extracted from ``h_n`` and
    returned as the fixed-length temporal embedding.

    For BiLSTM the output dimension is doubled (forward + backward concat).
    For a 'dense' layer type, an ``nn.Sequential`` stack of Linear layers is used,
    with the configured ``activation`` applied between layers.

    Layers are built lazily on the first ``forward()`` call because ``input_size``
    is not known until the input tensor is seen.
    """

    def __init__(self, layer_config: Dict[str, Any], name: str = "rnn_block", **kwargs) -> None:
        """
        Initialise RnnBlock from a configuration dictionary.

        :param layer_config: Dict with 'general' (layers, layer_type, dropout_rate)
            and 'parameters' (units, activation, recurrent_activation, initialisers).
        :param name: human-readable layer name for logging / serialization.
        """
        super().__init__()

        self.layer_config: Dict[str, Any] = layer_config
        self.layer_name: str = name

        # --- General hyper-parameters ---
        self.num_layers: int | None = None
        self.layer_type: str | None = None
        self.dropout_rate: float | None = None
        self._unpack_configuration(config=layer_config.get("general"))

        # --- Layer-specific parameters ---
        self.units: int | None = None
        self.activation: str | None = None
        self.recurrent_activation: str | None = None
        self.kernel_initializer: str | None = None
        self.recurrent_initializer: str | None = None
        self.bias_initializer: str | None = None
        self._unpack_configuration(config=layer_config.get("parameters"))

        # Lazy-build flag: layers are created on first forward() call
        self._built: bool = False
        self.rnn: nn.Module | None = None
        self.dense: nn.Module | None = None

    # ------------------------------------------------------------------
    # Layer construction (lazy, triggered on first forward)
    # ------------------------------------------------------------------

    def _build(self, input_size: int) -> None:
        """
        Create the recurrent (or dense) sub-module once ``input_size`` is known.

        :param input_size: last dimension of the input tensor (number of features).
        """
        layer_type = self.layer_type.lower()
        dropout = self.dropout_rate if self.num_layers > 1 else 0.0

        if layer_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=self.units,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif layer_type == "bilstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=self.units,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
        elif layer_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=self.units,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif layer_type == "dense":
            act_mod = _resolve_act_module(self.activation)
            layers = []
            in_dim = input_size
            for _ in range(self.num_layers):
                layers.append(nn.Linear(in_features=in_dim, out_features=self.units))
                if act_mod is not None:
                    layers.append(_resolve_act_module(self.activation))
                in_dim = self.units
            if act_mod is not None and len(layers) > 1:
                layers.pop()
            self.dense = nn.Sequential(*layers)
        else:
            raise ValueError(f"Undefined layer type, got {layer_type}")

        self._built = True

        if self.rnn is not None:
            self.rnn.train(self.training)
        if self.dense is not None:
            self.dense.train(self.training)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: P [B, S, T_e] -> r [B, d_r].

        :param inputs: P&L history tensor with shape [batch, sequence_length, num_features].
        :returns: Temporal embedding with shape [batch, units] (or [batch, 2*units] for BiLSTM).
        """
        if not self._built:
            input_size = inputs.size(-1)
            self._build(input_size)
            self.to(inputs.device)

        layer_type = self.layer_type.lower()

        if layer_type in _RECURRENT_TYPES:
            output, hidden = self.rnn(inputs)

            if layer_type in {"lstm", "bilstm"}:
                h_n = hidden[0]
            else:
                h_n = hidden

            if layer_type == "bilstm":
                h_forward = h_n[-2]
                h_backward = h_n[-1]
                x = torch.cat([h_forward, h_backward], dim=-1)  # [B, 2*units]
            else:
                x = h_n[-1]  # [B, units]

        elif layer_type == "dense":
            x = self.dense(inputs[:, -1, :])  # [B, units]

        else:
            raise ValueError(f"Undefined layer type, got {layer_type}")

        return x

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration dictionary for serialization."""
        return {"layer_config": self.layer_config, "name": self.layer_name}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RnnBlock":
        """Reconstruct an RnnBlock from a configuration dictionary.

        :param config: dict produced by ``get_config()``.
        :returns: new ``RnnBlock`` instance (un-built; weights not restored).
        """
        return cls(**config)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from a config sub-dict.

        The 'general' section uses 'layers' as the key; we store it as ``num_layers``
        to align with the PyTorch LSTM/GRU constructor parameter name.

        :param config: dictionary of key-value pairs to set as attributes.
        """
        for k, v in config.items():
            attr_name = "num_layers" if k == "layers" else k
            setattr(self, attr_name, v)
