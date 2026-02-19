"""
Data pipeline configuration for the GNN-RNN hybrid model.

``HybridGnnRnnDataConfig`` inherits ``DataPipelineConfig`` (universal pipeline
settings) and adds model-specific parameters for graph construction, PnL
history generation, and portfolio composition.

Usage:
    from src.machine_learning.data.gnn_rnn_hybrid.config import HybridGnnRnnDataConfig

    cfg = HybridGnnRnnDataConfig(
        batch_size=32,
        use_synthetic=True,
        n_trades=50,
        n_timesteps=20,
    )
    result = build_gnn_data(cfg)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.machine_learning.core.config import DataPipelineConfig


@dataclass
class HybridGnnRnnDataConfig(DataPipelineConfig):
    """
    Data configuration for the GNN-RNN hybrid model.

    Inherits universal pipeline settings (batch_size, shuffle, cache, etc.)
    from ``DataPipelineConfig`` and adds GNN-specific parameters.

    Attributes
    ----------
    use_synthetic : bool
        If True, generate synthetic data; else build from FX portfolio.
    seed : int, optional
        Random seed for reproducibility.

    Split ratios:
    train_ratio : float
        Fraction for training.
    val_ratio : float
        Fraction for validation.
    projection_ratio : float
        Fraction for projection / holdout.

    Synthetic data parameters (used when ``use_synthetic=True``):
    n_trades : int
        Number of trades in the portfolio graph.
    n_elementary : int
        Number of elementary (input) instruments.
    n_targets : int
        Number of target instruments to predict.
    n_samples : int
        Number of PnL scenarios.
    n_timesteps : int
        Time steps per scenario.
    k_neighbours : int
        K-nearest neighbours for adjacency matrix construction.
    noise_std : float
        Noise standard deviation for synthetic PnL generation.

    FX portfolio parameters (used when ``use_synthetic=False``):
    n_vanilla, n_digital, n_barrier, n_double_barrier, n_asian, n_touch : int
        Number of each instrument type in the FX portfolio.
    spot : float
        FX spot rate.
    sigma : float
        FX implied volatility.
    """

    use_synthetic: bool = True
    seed: Optional[int] = None

    # Split ratios
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    projection_ratio: float = 0.2

    # Synthetic data params
    n_trades: int = 50
    n_elementary: int = 30
    n_targets: int = 10
    n_samples: int = 500
    n_timesteps: int = 20
    k_neighbours: int = 5
    noise_std: float = 0.5

    # FX portfolio params
    n_vanilla: int = 100
    n_digital: int = 100
    n_barrier: int = 5
    n_double_barrier: int = 5
    n_asian: int = 5
    n_touch: int = 5
    spot: float = 1.10
    sigma: float = 0.15
