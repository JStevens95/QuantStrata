"""
Configuration schema and defaults for models/gnn_rnn_hybrid (HybridGnnRnn).
"""

from __future__ import annotations

from typing import Any, Dict


def default_hybrid_model_config(
    gnn_units: int = 32,
    rnn_units: int = 32,
    fusion_units: int = 32,
    attention_units: int = 32,
    projection_units: int = 32,
    n_targets: int = 10,
) -> Dict[str, Any]:
    """
    Return a minimal valid model_config for HybridGnnRnn.

    Parameters
    ----------
    gnn_units, rnn_units, fusion_units, attention_units, projection_units : int
        Hidden units for each block.
    n_targets : int
        Number of target trades (for projection baseline_trade_count).

    Returns
    -------
    dict
        model_config ready for HybridGnnRnn(model_config=...).
    """
    return {
        "general": {
            "architecture": "default",
        },
        "gnn_model": {
            "general": {
                "layers": 2,
                "layer_type": "graph_sage",
                "dropout_rate": 0.1,
                "use_bias": True,
                "use_residual": True,
                "layer_norm": True,
            },
            "parameters": {
                "units": gnn_units,
                "activation": "relu",
                "kernel_initializer": "glorot_uniform",
                "bias_initializer": "zeros",
            },
        },
        "rnn_model": {
            "general": {
                "layers": 2,
                "layer_type": "lstm",
                "dropout_rate": 0.1,
            },
            "parameters": {
                "units": rnn_units,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "kernel_initializer": "glorot_uniform",
                "recurrent_initializer": "orthogonal",
                "bias_initializer": "zeros",
            },
        },
        "fusion_model": {
            "general": {
                "dropout_rate": 0.1,
                "fusion_mode": "gate",
                "num_heads": 2,
            },
            "parameters": {
                "units": fusion_units,
                "activation": "relu",
                "kernel_initializer": "glorot_uniform",
                "bias_initializer": "zeros",
            },
        },
        "attention_model": {
            "general": {
                "dropout_rate": 0.1,
                "num_heads": 2,
            },
            "parameters": {
                "units": attention_units,
                "activation": "relu",
                "kernel_initializer": "glorot_uniform",
                "bias_initializer": "zeros",
            },
        },
        "projection_model": {
            "general": {
                "baseline_new_mode": "output_mix",
                "baseline_trade_count": n_targets,
                "dropout_rate": 0.1,
                "new_target_mode": "knn",
                "use_baseline_norm": True,
                "use_attn_scale": True,
                "use_attn_bias": True,
                "residual_new_damp": 1.0,
                "knn_k": 4,
                "knn_power": 2.0,
                "knn_temperature": 5.0,
                "knn_mode": "cosine_softmax",
                "knn_eps": 1e-8,
            },
            "parameters": {
                "units": projection_units,
                "activation": None,
                "kernel_initializer": "glorot_uniform",
                "bias_initializer": "zeros",
            },
        },
    }


__all__ = ["default_hybrid_model_config"]
