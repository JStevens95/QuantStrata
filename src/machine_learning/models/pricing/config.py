"""
Configuration schema and defaults for models/pricing (MLP pricer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PricingModelConfig:
    """
    Default configuration for MLPPricer.

    Attributes
    ----------
    n_features : int
        Input feature dimension (e.g. 6 for spot, strike, vol, rate, expiry, is_call).
    hidden_units : list of int
        Hidden layer sizes.
    activation : str
        Activation function name.
    dropout_rate : float
        Dropout rate (0 to disable).
    use_batch_norm : bool
        Whether to use batch normalization.
    use_skip_connections : bool
        Whether to use residual connections.
    kernel_regularizer : float
        L2 regularization weight (0 to disable).
    """

    n_features: int = 6
    hidden_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    use_skip_connections: bool = False
    kernel_regularizer: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for logging or persistence."""
        return {
            "n_features": self.n_features,
            "hidden_units": self.hidden_units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "use_skip_connections": self.use_skip_connections,
            "kernel_regularizer": self.kernel_regularizer,
        }


def default_pricing_config(
    n_features: int = 6,
    hidden_units: List[int] = None,
    dropout_rate: float = 0.1,
) -> PricingModelConfig:
    """Return a default config for MLPPricer."""
    if hidden_units is None:
        hidden_units = [128, 64, 32]
    return PricingModelConfig(
        n_features=n_features,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
    )


__all__ = ["PricingModelConfig", "default_pricing_config"]
