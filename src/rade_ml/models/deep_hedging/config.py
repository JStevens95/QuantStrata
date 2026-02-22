"""
Configuration scheme and defaults for the Deep Hedging model.

The config is split into four sections:
    - general:      global model settings (instruments, costs, constraints)
    - encoder:      GatedResidualNetwork settings for per-timestep feature encoding
    - policy:       recurrent hedging policy network settings
    - risk_measure: loss function / risk measure configuration
"""
from __future__ import annotations

from typing import Any, Dict


def default_model_config() -> Dict[str, Any]:
    """Returns a minimal valid model configuration for DeepHedgingModel."""
    return {
        "general": {
            "num_hedging_instruments": 1,
            "transaction_cost_rate": 0.001,
            "position_limit": None,
        },
        "encoder": {
            "units": 64,
            "dropout_rate": 0.1,
            "activation": "elu",
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
        },
        "policy": {
            "rnn_type": "gru",
            "rnn_units": 128,
            "rnn_layers": 2,
            # dropout_rate=0: GRUCell dropout causes InaccessibleTensorError when policy.step()
            # is called in a loop inside tf.function. Use 0 for graph-mode compatibility.
            "dropout_rate": 0.0,
            "output_activation": None,
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "orthogonal",
            "bias_initializer": "zeros",
        },
        "risk_measure": {
            "type": "cvar",
            "alpha": 0.95,
            "lagrange_multiplier": 1.0,
        },
    }
