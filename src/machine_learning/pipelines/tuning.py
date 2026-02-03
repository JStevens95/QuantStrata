"""
Hyperparameter tuning pipeline for QuantStrata ML models.

Provides run_tuning() for grid or random search over hyperparameters,
returning a standardised TuningResult for reproducibility and comparison.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from src.machine_learning.core.types import TuningResult

logger = logging.getLogger(__name__)


def _grid_search(
    objective_fn: Callable[[Dict[str, Any]], float],
    search_space: Dict[str, Sequence[Any]],
) -> List[Dict[str, Any]]:
    """Generate all trial configs from a grid (Cartesian product)."""
    keys = list(search_space.keys())
    if not keys:
        return [{}]

    head, rest = keys[0], keys[1:]
    rest_configs = _grid_search(
        objective_fn,
        {k: search_space[k] for k in rest},
    ) if rest else [{}]

    trials = []
    for v in search_space[head]:
        for r in rest_configs:
            trials.append({head: v, **r})
    return trials


def run_tuning(
    objective_fn: Callable[[Dict[str, Any]], float],
    search_space: Dict[str, Sequence[Any]],
    method: str = "grid",
    n_trials: Optional[int] = None,
    minimize: bool = True,
    checkpoint_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TuningResult:
    """
    Run hyperparameter search and return a standardised TuningResult.

    Parameters
    ----------
    objective_fn : callable
        Function that takes a config dict and returns a scalar score.
        Lower is better if minimize=True, else higher is better.
    search_space : dict
        For grid: each key maps to a sequence of values (e.g. {"lr": [1e-3, 1e-2], "units": [32, 64]}).
        For random: each key maps to a sequence (n_trials configs sampled uniformly).
    method : str
        "grid" (exhaustive) or "random" (sample n_trials).
    n_trials : int, optional
        For method="random", number of trials to run. Required for random.
    minimize : bool
        If True, best = smallest score; if False, best = largest score.
    checkpoint_dir : str, optional
        If provided, objective_fn will receive it in config for saving best checkpoint.
    metadata : dict, optional
        Additional metadata for the tuning result.

    Returns
    -------
    TuningResult
        best_config, best_score, trials, metadata.

    Example
    -------
    >>> def train_and_score(config):
    ...     model = create_model(units=config["units"])
    ...     model.fit(..., epochs=config["epochs"])
    ...     return validation_loss
    >>> result = run_tuning(
    ...     train_and_score,
    ...     search_space={"units": [32, 64], "epochs": [10, 20]},
    ...     method="grid",
    ...     minimize=True,
    ... )
    >>> print(result.best_config, result.best_score)
    """
    import random

    if method == "grid":
        trial_configs = _grid_search(objective_fn, search_space)
    elif method == "random":
        if n_trials is None:
            raise ValueError("n_trials is required for method='random'")
        trial_configs = []
        keys = list(search_space.keys())
        for _ in range(n_trials):
            trial_configs.append({
                k: random.choice(list(search_space[k])) for k in keys
            })
    else:
        raise ValueError(f"method must be 'grid' or 'random', got {method!r}")

    trials: List[Dict[str, Any]] = []
    best_score: float = float("inf") if minimize else float("-inf")
    best_config: Dict[str, Any] = {}

    for i, config in enumerate(trial_configs):
        if checkpoint_dir:
            config = {**config, "checkpoint_dir": checkpoint_dir}
        try:
            score = objective_fn(config)
        except Exception as e:
            logger.warning("Trial %s failed: %s", config, e)
            trials.append({"config": config, "score": None, "metadata": {"error": str(e)}})
            continue

        trials.append({"config": config, "score": score, "metadata": {}})
        if minimize and score < best_score:
            best_score = score
            best_config = config.copy()
        elif not minimize and score > best_score:
            best_score = score
            best_config = config.copy()

    return TuningResult(
        best_config=best_config,
        best_score=best_score,
        best_checkpoint_path=None,
        trials=trials,
        metadata={
            "method": method,
            "n_trials": len(trial_configs),
            **(metadata or {}),
        },
    )


__all__ = ["run_tuning", "_grid_search"]
