"""
Configuration scheme and defaults for HybridGnnRnn model.
"""
from __future__ import annotations

from typing import Any, Dict


def default_model_config() -> Dict[str, Any]:
    """Returns a minimal valid model configuration for HybridGnnRnn model."""
    return {
        "general": {

        },
        "gnn_layer": {
            "general": {
                "architecture": "default",
                "layers": 2,
                "layer_type": "mixed_graph_sage",
                "dropout_rate": 0.1,
                "use_bias": True,
                "use_residual": True,
                "batch_norm": True,
                "aggregator_op": "mean",
            },
            "parameters": {
                "units": 128,
                "activation": "relu",
                "kernel_initializer": "glorot_uniform",
                "bias_initializer": "zeros",
            }
        },
        "rnn_layer": {
            "general": {
                "architecture": "default",
                "layers": 2,
                "layer_type": "lstm",
                "dropout_rate": 0.1,
                "use_bias": True,
            },
            "parameters": {
                "units": 128,
                "activation": "relu",
                "recurrent_activation": "sigmoid",
                "kernel_initializer": "glorot_uniform",
                "recurrent_initializer": "orthogonal",
                "bias_initializer": "zeros",
            }
        },
        "fusion_layer": {
            "general": {
                "fusion_mode": "gate",
                "dropout_rate": 0.1,
                "num_heads": 1,
                "k_nbrs": 50,
            },
            "parameters": {
                "units": 64,
                "activation": "sigmoid",
                "kernel_initializer": "he_uniform",
                "bias_initializer": "zeros",
            }
        },
        "attention_layer": {
            "general": {
                "layer_type": "standard",
                "use_residual": True,
                "use_layer_norm": True,
                "attention_mode": True,
                "num_heads": 1,
                "dropout_rate": 0.1,
                "k_nbrs": 50,
            },
            "parameters": {
                "units": 32,
                "activation": "tanh",
                "kernel_initializer": "he_uniform",
                "bias_initializer": "zeros",
            }
        },
        "projection_layer": {
            "general": {
                "dropout_rate": 0.1,
                "baseline_new_mode": "output_mix",
                "use_baseline_norm": True,
                "use_attn_scale_new": False,
                "use_attn_bias_new": False,
                "knn_k": 5,
                "knn_mode": "cosine_softmax",
                "knn_temperature": 5.0,
                "knn_power": 2.0,
                "residual_new_damp": 1.0
            },
            "parameters": {
                "units": 32,
                "activation": "gelu",
                "kernel_initializer": "glorot_uniform",
                "bias_initializer": "zeros",
            }
        }
    }
