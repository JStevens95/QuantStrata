"""
Configuration schema and defaults for models/gnn_rnn_hybrid (HybridGnnRnn).

Re-exports default_hybrid_model_config from data layer so model package
exposes a single place for model config.
"""

from __future__ import annotations

from typing import Any, Dict

from src.m_learning.data.gnn_synthetic import default_hybrid_model_config as _default_hybrid_model_config


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
    return _default_hybrid_model_config(
        gnn_units=gnn_units,
        rnn_units=rnn_units,
        fusion_units=fusion_units,
        attention_units=attention_units,
        projection_units=projection_units,
        n_targets=n_targets,
    )


__all__ = ["default_hybrid_model_config"]
