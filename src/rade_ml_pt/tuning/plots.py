"""
Tuning-specific visualisations.

Wraps Optuna's built-in plotting utilities with consistent styling and
provides additional analytics views.  All functions accept a ``TuningResult``
or a raw ``optuna.Study`` for flexibility.

Functions
---------
plot_optimization_history
    Objective value over trials.
plot_param_importances
    Fanova / mean-decrease importance of each hyperparameter.
plot_parallel_coordinate
    Multi-dimensional view of parameters coloured by objective.
plot_contour
    2-D contour landscape for a selected pair of parameters.
plot_slice
    Marginal objective value vs. each parameter.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna

from src.rade_ml_pt.core.types import TuningResult

logger = logging.getLogger(__name__)


def _get_study(source: Union[TuningResult, optuna.Study]) -> optuna.Study:
    """Resolve the underlying Optuna study from the input."""
    if isinstance(source, optuna.Study):
        return source
    raise TypeError(
        "Tuning plots require a live optuna.Study instance.  "
        "Pass `tuner.study` rather than the serialised TuningResult."
    )


# ------------------------------------------------------------------
# Matplotlib-native plots (work from TuningResult without live study)
# ------------------------------------------------------------------

def plot_optimization_history(
    result: TuningResult,
    ax: Optional[plt.Axes] = None,
    show_best: bool = True,
) -> plt.Axes:
    """
    Plot objective value over trials with a running-best line.

    Parameters
    ----------
    result : TuningResult
        Completed tuning result.
    ax : matplotlib Axes, optional
        Axes to draw on. Created if None.
    show_best : bool
        If True, overlay a running-best line.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    completed = [t for t in result.all_trials if t["state"] == "COMPLETE" and t["value"] is not None]
    if not completed:
        logger.warning("No completed trials to plot.")
        return ax

    numbers = [t["number"] for t in completed]
    values = [t["value"] for t in completed]

    ax.scatter(numbers, values, alpha=0.6, s=20, label="Trial value", zorder=3)

    if show_best:
        if result.direction == "minimize":
            running_best = np.minimum.accumulate(values)
        else:
            running_best = np.maximum.accumulate(values)
        ax.plot(numbers, running_best, color="crimson", linewidth=1.5, label="Best so far")

    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.set_title("Optimisation History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_param_importances(
    study: optuna.Study,
    ax: Optional[plt.Axes] = None,
    top_n: int = 10,
) -> plt.Axes:
    """
    Horizontal bar chart of hyperparameter importances.

    Uses Optuna's fANOVA-based importance evaluator internally.

    Parameters
    ----------
    study : optuna.Study
        Live Optuna study (needs raw trial data).
    ax : matplotlib Axes, optional
    top_n : int
        Maximum number of params to display.

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    importances = optuna.importance.get_param_importances(study)
    params = list(importances.keys())[:top_n]
    values = [importances[p] for p in params]

    y_pos = np.arange(len(params))
    ax.barh(y_pos, values, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Hyperparameter Importances")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_parallel_coordinate(
    study: optuna.Study,
    params: Optional[List[str]] = None,
) -> None:
    """
    Multi-dimensional parallel-coordinate plot via Optuna's matplotlib backend.

    Parameters
    ----------
    study : optuna.Study
    params : list of str, optional
        Subset of parameter names to include. All params if None.
    """
    _ = optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=params)
    plt.title("Parallel Coordinate Plot")
    plt.tight_layout()


def plot_contour(
    study: optuna.Study,
    params: List[str],
) -> None:
    """
    2-D contour plot of the objective landscape for a pair of parameters.

    Parameters
    ----------
    study : optuna.Study
    params : list of str
        Exactly two parameter names.
    """
    if len(params) != 2:
        raise ValueError("plot_contour requires exactly two parameter names.")

    _ = optuna.visualization.matplotlib.plot_contour(study, params=params)
    plt.title(f"Contour: {params[0]} vs {params[1]}")
    plt.tight_layout()


def plot_slice(
    study: optuna.Study,
    params: Optional[List[str]] = None,
) -> None:
    """
    Marginal objective value vs. each parameter (slice plot).

    Parameters
    ----------
    study : optuna.Study
    params : list of str, optional
        Subset of parameters. All if None.
    """
    _ = optuna.visualization.matplotlib.plot_slice(study, params=params)
    plt.suptitle("Slice Plot", y=1.02)
    plt.tight_layout()
