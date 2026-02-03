"""
Synthetic data generation for GNN-RNN hybrid model.

Generates trade features, k-NN adjacency, PnL history, and targets
for the HybridGnnRnn model. Used by data/gnn_rnn_hybrid/build.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


@dataclass
class SyntheticGnnData:
    """
    Container for synthetic GNN-LSTM data.

    Attributes
    ----------
    trade_features : np.ndarray
        Shape (n_trades, n_features). Trade attribute vectors.
    adjacency_matrix : np.ndarray
        Shape (n_trades, n_trades). Row-normalised k-NN adjacency.
    pnl_history : np.ndarray
        Shape (n_samples, n_timesteps, n_elementary). PnL time series.
    targets : np.ndarray
        Shape (n_samples, n_targets). Target PnL values.
    elementary_indices : np.ndarray
        Shape (n_elementary,). Indices of elementary trades.
    target_indices : np.ndarray
        Shape (n_targets,). Indices of target trades.
    feature_names : list[str]
        Names of the features in trade_features.
    """

    trade_features: np.ndarray
    adjacency_matrix: np.ndarray
    pnl_history: np.ndarray
    targets: np.ndarray
    elementary_indices: np.ndarray
    target_indices: np.ndarray
    feature_names: list

    def to_gnn_inputs(self) -> Dict[str, np.ndarray]:
        """Package into dict expected by HybridGnnRnn."""
        return {
            "trade_features": self.trade_features.astype(np.float32),
            "adjacency_matrix": self.adjacency_matrix.astype(np.float32),
            "pnl_history": self.pnl_history.astype(np.float32),
            "target_indices": self.target_indices.astype(np.int32),
            "elementary_indices": self.elementary_indices.astype(np.int32),
        }


def generate_synthetic_trade_features(
    n_trades: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, list]:
    """
    Generate synthetic trade feature matrix.

    Features: moneyness, time_to_maturity, delta, vega, product_type (one-hot 3 classes).

    Parameters
    ----------
    n_trades : int
        Number of trades.
    seed : int, optional
        Random seed.

    Returns
    -------
    features : np.ndarray
        Shape (n_trades, 7).
    feature_names : list
        Names of the 7 features.
    """
    rng = np.random.default_rng(seed)
    moneyness = rng.uniform(0.8, 1.2, n_trades)
    ttm = rng.uniform(0.1, 2.0, n_trades)
    delta = rng.uniform(-1.0, 1.0, n_trades)
    vega = rng.uniform(0.0, 50.0, n_trades)
    prod_idx = rng.integers(0, 3, n_trades)
    prod_onehot = np.eye(3)[prod_idx]

    features = np.column_stack([moneyness, ttm, delta, vega, prod_onehot])
    feature_names = [
        "moneyness",
        "time_to_maturity",
        "delta",
        "vega",
        "prod_type_0",
        "prod_type_1",
        "prod_type_2",
    ]
    return features.astype(np.float32), feature_names


def build_knn_adjacency(
    features: np.ndarray,
    k: int = 5,
    include_self: bool = True,
) -> np.ndarray:
    """
    Build row-normalised k-NN adjacency matrix from feature matrix.

    Parameters
    ----------
    features : np.ndarray
        Shape (n, d).
    k : int
        Number of neighbours (excluding self unless include_self=True).
    include_self : bool
        Whether to add self-loops.

    Returns
    -------
    adj : np.ndarray
        Shape (n, n). Row-normalised adjacency.
    """
    n = features.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    rows = []
    cols = []
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            if j == i and not include_self:
                continue
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    adj_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))

    row_sums = np.array(adj_sparse.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    adj_dense = adj_sparse.toarray() / row_sums[:, None]
    return adj_dense.astype(np.float32)


def generate_pnl_history(
    n_samples: int,
    n_timesteps: int,
    n_elementary: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic PnL history (random walk). Shape (n_samples, n_timesteps, n_elementary)."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, 1.0, (n_samples, n_timesteps, n_elementary))
    pnl = np.cumsum(increments, axis=1)
    return pnl.astype(np.float32)


def generate_targets(
    pnl_history: np.ndarray,
    elementary_indices: np.ndarray,
    target_indices: np.ndarray,
    noise_std: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate target PnL as linear combination of final elementary PnL + noise. Shape (n_samples, n_targets)."""
    rng = np.random.default_rng(seed)
    n_samples = pnl_history.shape[0]
    n_elementary = pnl_history.shape[2]
    n_targets = len(target_indices)

    final_pnl = pnl_history[:, -1, :]
    weights = rng.uniform(0.1, 1.0, (n_elementary, n_targets))
    weights = weights / weights.sum(axis=0, keepdims=True)
    targets = final_pnl @ weights + rng.normal(0.0, noise_std, (n_samples, n_targets))
    return targets.astype(np.float32)


def generate_synthetic_gnn_data(
    n_trades: int = 50,
    n_elementary: int = 30,
    n_targets: int = 10,
    n_samples: int = 500,
    n_timesteps: int = 20,
    k_neighbours: int = 5,
    noise_std: float = 0.5,
    seed: Optional[int] = None,
) -> SyntheticGnnData:
    """
    Generate a complete synthetic dataset for GNN-RNN hybrid model.

    Returns
    -------
    SyntheticGnnData
        Container with all arrays.
    """
    rng = np.random.default_rng(seed)

    features, feature_names = generate_synthetic_trade_features(n_trades, seed=seed)
    adjacency = build_knn_adjacency(features, k=k_neighbours, include_self=True)

    all_indices = np.arange(n_trades)
    rng.shuffle(all_indices)
    elementary_indices = np.sort(all_indices[:n_elementary])
    target_indices = np.sort(all_indices[n_elementary : n_elementary + n_targets])

    if n_elementary + n_targets > n_trades:
        raise ValueError("n_elementary + n_targets must be <= n_trades")

    pnl_history = generate_pnl_history(n_samples, n_timesteps, n_elementary, seed=seed)
    targets = generate_targets(
        pnl_history, elementary_indices, target_indices, noise_std, seed=seed
    )

    return SyntheticGnnData(
        trade_features=features,
        adjacency_matrix=adjacency,
        pnl_history=pnl_history,
        targets=targets,
        elementary_indices=elementary_indices,
        target_indices=target_indices,
        feature_names=feature_names,
    )


__all__ = [
    "SyntheticGnnData",
    "generate_synthetic_trade_features",
    "build_knn_adjacency",
    "generate_pnl_history",
    "generate_targets",
    "generate_synthetic_gnn_data",
]
