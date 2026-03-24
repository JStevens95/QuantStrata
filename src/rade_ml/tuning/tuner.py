"""
Model-independent hyperparameter tuner backed by Optuna.

The ``Tuner`` manages study creation, trial execution, pruning, and
result aggregation.  The caller supplies an objective function that takes
an ``optuna.Trial`` and returns a scalar score; internally the objective
can use the generic ``Trainer`` to build and train any model.

Usage::

    from rade_ml.tuning import Tuner

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        units = trial.suggest_int("gnn_units", 32, 256, step=32)
        # ... build model, train, return val_loss
        return val_loss

    tuner = Tuner(n_trials=50, direction="minimize")
    result = tuner.run(objective)
"""
from __future__ import annotations

import logging
import time

from typing import Any, Callable, Dict, List, Optional

import optuna

from src.rade_ml.core.types import TuningResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pruner factory
# ------------------------------------------------------------------

_PRUNER_MAP = {
    "median": optuna.pruners.MedianPruner,
    "percentile": optuna.pruners.PercentilePruner,
    "hyperband": optuna.pruners.HyperbandPruner,
    "threshold": optuna.pruners.ThresholdPruner,
    "none": optuna.pruners.NopPruner,
}


def _build_pruner(
    name: str,
    pruner_kwargs: Optional[Dict[str, Any]] = None,
) -> optuna.pruners.BasePruner:
    """Instantiate a pruner by name."""
    key = name.lower()
    if key not in _PRUNER_MAP:
        raise ValueError(f"Unknown pruner '{name}'. Choose from: {list(_PRUNER_MAP.keys())}")
    return _PRUNER_MAP[key](**(pruner_kwargs or {}))


# ------------------------------------------------------------------
# Tuner
# ------------------------------------------------------------------

class Tuner:
    """
    Optuna-backed hyperparameter tuner.

    Parameters
    ----------
    n_trials : int
        Number of trials to run.
    direction : str
        "minimize" or "maximize".
    pruner : str
        Pruner strategy name (see ``_PRUNER_MAP``).
    pruner_kwargs : dict, optional
        Extra kwargs forwarded to the pruner constructor.
    seed : int or None
        Random seed for the sampler.
    study_name : str or None
        Name for the Optuna study. Auto-generated if None.
    storage : str or None
        Optuna storage URL (e.g. ``sqlite:///study.db``).  Defaults to
        in-memory storage.
    sampler_kwargs : dict, optional
        Extra kwargs forwarded to the TPE sampler.
    """

    def __init__(
        self,
        n_trials: int = 50,
        direction: str = "minimize",
        pruner: str = "median",
        pruner_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_trials = n_trials
        self.direction = direction
        self.seed = seed

        sampler_kw: Dict[str, Any] = sampler_kwargs or {}
        if seed is not None:
            sampler_kw.setdefault("seed", seed)

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=_build_pruner(pruner, pruner_kwargs),
            sampler=optuna.samplers.TPESampler(**sampler_kw),
            storage=storage,
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        objective: Callable[[optuna.Trial], float],
        callbacks: Optional[List[Callable]] = None,
        show_progress_bar: bool = True,
    ) -> TuningResult:
        """
        Execute the hyperparameter search.

        Parameters
        ----------
        objective : callable
            Function that accepts an ``optuna.Trial`` and returns a scalar.
        callbacks : list of callable, optional
            Optuna callbacks invoked after each trial.
        show_progress_bar : bool
            Display a tqdm progress bar during optimisation.

        Returns
        -------
        TuningResult
        """
        t0 = time.perf_counter()

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=callbacks,
            show_progress_bar=show_progress_bar,
        )

        elapsed = time.perf_counter() - t0
        result = self._build_result(elapsed)

        logger.info(
            f"Tuning complete: {result.n_completed}/{result.n_trials} trials, "
            f"best={result.best_value:.6f} (trial {result.best_trial_number})"
        )
        return result

    @property
    def best_trial(self) -> Dict[str, Any]:
        """Return the best trial's params and value."""
        bt = self.study.best_trial
        return {
            "number": bt.number,
            "value": bt.value,
            "params": bt.params,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(self, elapsed: float) -> TuningResult:
        """Aggregate study results into a TuningResult."""
        trials_summary = []
        n_completed = 0
        n_pruned = 0

        for t in self.study.trials:
            summary: Dict[str, Any] = {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
                "duration_seconds": (
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    if t.datetime_complete and t.datetime_start
                    else None
                ),
            }
            trials_summary.append(summary)

            if t.state == optuna.trial.TrialState.COMPLETE:
                n_completed += 1
            elif t.state == optuna.trial.TrialState.PRUNED:
                n_pruned += 1

        best = self.study.best_trial

        return TuningResult(
            study_name=self.study.study_name or "",
            direction=self.direction,
            n_trials=len(self.study.trials),
            n_completed=n_completed,
            n_pruned=n_pruned,
            best_trial_number=best.number,
            best_value=best.value,
            best_params=best.params,
            all_trials=trials_summary,
            elapsed_seconds=elapsed,
        )
