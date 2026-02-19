"""
GNN-RNN hybrid model data builder.

Produces train / val / projection ``tf.data.Dataset`` splits for
``models/gnn_rnn_hybrid``.  Splitting uses ``sklearn.model_selection.train_test_split``;
pipeline settings come from ``HybridGnnRnnDataConfig``.

Contract:
    ``build_gnn_data(config)`` returns a ``GnnDataResult`` (subclass of ``DataBuildResult``)
    with ``train_ds``, ``val_ds``, ``proj_ds`` so the training pipeline has a single,
    repeatable interface.

Usage:
    from src.machine_learning.data.gnn_rnn_hybrid.config import HybridGnnRnnDataConfig
    from src.machine_learning.data.gnn_rnn_hybrid.build import build_gnn_data

    cfg = HybridGnnRnnDataConfig(batch_size=32, n_samples=500, seed=42)
    result = build_gnn_data(cfg)
    trainer.fit(result.train_ds, result.val_ds)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.machine_learning.data.dataset import build_tf_dataset
from src.machine_learning.data.result import DataBuildResult
from src.machine_learning.data.gnn_rnn_hybrid.config import HybridGnnRnnDataConfig
from src.machine_learning.data.gnn_rnn_hybrid.synthetic import generate_synthetic_gnn_data
from src.machine_learning.data.gnn_rnn_hybrid.portfolio_builder import (
    build_fx_gnn_data,
    train_val_projection_split,
)


@dataclass
class GnnDataResult(DataBuildResult):
    """
    Result of ``build_gnn_data``: ``tf.data.Dataset`` splits for GNN-RNN hybrid.

    Attributes
    ----------
    train_ds : tf.data.Dataset
        Training dataset (batched, shuffled).
    val_ds : tf.data.Dataset
        Validation dataset (batched, no shuffle).
    proj_ds : tf.data.Dataset
        Projection / holdout dataset (batched, no shuffle).
    metadata : dict
        Data build metadata (n_trades, n_samples, splits, etc.).
    """

    proj_ds: Any = None  # tf.data.Dataset (projection / holdout)

    @property
    def holdout_ds(self) -> Any:
        """Alias for proj_ds so pipeline can use result.holdout_ds generically."""
        return self.proj_ds


def _train_val_proj_split(
    indices: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    projection_ratio: float,
    random_state: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices into train / val / projection via sklearn.train_test_split.

    Uses two train_test_split calls: first separates train from (val+proj),
    then splits (val+proj) into val and proj.
    """
    test_size = val_ratio + projection_ratio
    train_idx, tmp_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    rel_proj = projection_ratio / test_size if test_size > 0 else 0.0
    val_idx, proj_idx = train_test_split(
        tmp_idx, test_size=rel_proj, random_state=random_state
    )
    return train_idx, val_idx, proj_idx


def _extract_variable_inputs(
    pnl: np.ndarray,
    targets: np.ndarray,
    elementary_indices: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Build variable_inputs dict and targets from sliced PnL and targets.

    Handles both 2D pnl (n_samples, n_instruments) and 3D pnl (n_samples, n_timesteps, n_instruments).
    """
    if pnl.ndim == 3:
        elem_pnl = pnl[:, :, elementary_indices].astype(np.float32)
    else:
        elem_pnl = pnl[:, elementary_indices].astype(np.float32)
    variable = {"elem_pnl_history": elem_pnl}
    tgt = targets.astype(np.float32)
    return variable, tgt


def _get_static_inputs(data: Any) -> Dict[str, np.ndarray]:
    """Extract the static (non-batched) graph arrays from a GNN data object."""
    return {
        "trade_features": data.trade_features,
        "adjacency_matrix": data.adjacency_matrix,
        "target_indices": data.target_indices,
        "elementary_indices": data.elementary_indices,
    }


def _build_eval_dataset(
    variable_inputs: Dict[str, np.ndarray],
    targets: np.ndarray,
    static_inputs: Dict[str, np.ndarray],
    pipe_kwargs: Dict[str, Any],
) -> Any:
    """Build a tf.data.Dataset for val/proj (no shuffle)."""
    return build_tf_dataset(
        variable_inputs=variable_inputs,
        targets=targets,
        static_inputs=static_inputs,
        shuffle=False,
        batch_size=pipe_kwargs["batch_size"],
        cache=pipe_kwargs["cache"],
        ensure_float32=pipe_kwargs["ensure_float32"],
    )


def build_gnn_data(config: Optional[HybridGnnRnnDataConfig] = None) -> GnnDataResult:
    """
    Build GNN-RNN data and return train / val / projection ``tf.data.Dataset`` splits.

    Steps:
        1. Generate raw data (synthetic or FX portfolio).
        2. Split scenario indices via sklearn.train_test_split (two-stage for 3-way split).
        3. Extract variable inputs and targets per split; static graph arrays shared.
        4. Assemble each split via ``build_tf_dataset`` (injects static into every batch).

    Parameters
    ----------
    config : HybridGnnRnnDataConfig, optional
        Full data pipeline + model-specific configuration.
        Defaults to ``HybridGnnRnnDataConfig()`` if not provided.

    Returns
    -------
    GnnDataResult
        ``train_ds``, ``val_ds``, ``proj_ds``, and build metadata.
    """
    if config is None:
        config = HybridGnnRnnDataConfig()

    if abs(config.train_ratio + config.val_ratio + config.projection_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + projection_ratio must equal 1.0")

    # 1. Generate raw data
    if config.use_synthetic:
        data: Any = generate_synthetic_gnn_data(
            n_trades=config.n_trades,
            n_elementary=config.n_elementary,
            n_targets=config.n_targets,
            n_samples=config.n_samples,
            n_timesteps=config.n_timesteps,
            k_neighbours=config.k_neighbours,
            noise_std=config.noise_std,
            seed=config.seed,
        )
        metadata = {
            "source": "synthetic",
            "n_trades": config.n_trades,
            "n_elementary": config.n_elementary,
            "n_targets": config.n_targets,
            "n_samples": config.n_samples,
            "n_timesteps": config.n_timesteps,
            "k_neighbours": config.k_neighbours,
            "seed": config.seed,
        }
    else:
        data = build_fx_gnn_data(
            n_vanilla=config.n_vanilla,
            n_digital=config.n_digital,
            n_barrier=config.n_barrier,
            n_double_barrier=config.n_double_barrier,
            n_asian=config.n_asian,
            n_touch=config.n_touch,
            n_samples=config.n_samples,
            n_timesteps=config.n_timesteps,
            k_neighbours=config.k_neighbours,
            spot=config.spot,
            sigma=config.sigma,
            noise_std=config.noise_std,
            seed=config.seed,
        )
        train_inputs, train_targets, val_inputs, val_targets, proj_inputs, proj_targets = (
            train_val_projection_split(
                data,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                projection_ratio=config.projection_ratio,
            )
        )
        # FX path uses different structure; convert to our variable/static format
        static = _get_static_inputs(data)
        pipe_kwargs = config.to_build_kwargs()

        def _from_fx_inputs(inp: Dict, tgt: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
            var, _ = _extract_variable_inputs(
                inp["pnl_history"], tgt, data.elementary_indices
            )
            return var, tgt.astype(np.float32)

        train_var, train_tgt = _from_fx_inputs(train_inputs, train_targets)
        val_var, val_tgt = _from_fx_inputs(val_inputs, val_targets)
        proj_var, proj_tgt = _from_fx_inputs(proj_inputs, proj_targets)

        train_ds = build_tf_dataset(
            variable_inputs=train_var,
            targets=train_tgt,
            static_inputs=static,
            **pipe_kwargs,
        )
        val_ds = _build_eval_dataset(val_var, val_tgt, static, pipe_kwargs)
        proj_ds = _build_eval_dataset(proj_var, proj_tgt, static, pipe_kwargs)

        metadata = {
            "source": "fx_portfolio",
            "n_samples": config.n_samples,
            "n_timesteps": config.n_timesteps,
            "seed": config.seed,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "projection_ratio": config.projection_ratio,
            "pipeline": config.to_dict(),
        }
        return GnnDataResult(
            train_ds=train_ds,
            val_ds=val_ds,
            proj_ds=proj_ds,
            metadata=metadata,
        )

    # 2. Split scenario indices via sklearn
    n = data.pnl_history.shape[0]
    indices = np.arange(n)
    train_idx, val_idx, proj_idx = _train_val_proj_split(
        indices,
        config.train_ratio,
        config.val_ratio,
        config.projection_ratio,
        config.seed,
    )

    # 3. Extract variable inputs and targets per split
    train_var, train_tgt = _extract_variable_inputs(
        data.pnl_history[train_idx],
        data.targets[train_idx],
        data.elementary_indices,
    )
    val_var, val_tgt = _extract_variable_inputs(
        data.pnl_history[val_idx],
        data.targets[val_idx],
        data.elementary_indices,
    )
    proj_var, proj_tgt = _extract_variable_inputs(
        data.pnl_history[proj_idx],
        data.targets[proj_idx],
        data.elementary_indices,
    )

    # 4. Static inputs and assemble datasets
    static = _get_static_inputs(data)
    pipe_kwargs = config.to_build_kwargs()

    train_ds = build_tf_dataset(
        variable_inputs=train_var,
        targets=train_tgt,
        static_inputs=static,
        **pipe_kwargs,
    )
    val_ds = _build_eval_dataset(val_var, val_tgt, static, pipe_kwargs)
    proj_ds = _build_eval_dataset(proj_var, proj_tgt, static, pipe_kwargs)

    metadata.update({
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "projection_ratio": config.projection_ratio,
        "pipeline": config.to_dict(),
    })

    return GnnDataResult(
        train_ds=train_ds,
        val_ds=val_ds,
        proj_ds=proj_ds,
        metadata=metadata,
    )


__all__ = ["GnnDataResult", "build_gnn_data"]
