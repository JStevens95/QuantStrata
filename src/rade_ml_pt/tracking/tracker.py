"""
Lightweight experiment tracker backed by the local filesystem.

Each run is stored as ``store_dir/{run_id}/run.json``, making runs trivially
inspectable with any JSON viewer and easy to version-control.

Usage::

    tracker = ExperimentTracker("./experiments")
    run = tracker.start_run(name="gnn-rnn-cosine-lr", tags=["lr-sweep"])
    run.log_config(training_config)
    run.log_result(training_result)
    run.end()

    tracker.list_runs(tag="lr-sweep")
    tracker.compare_runs(["run_001", "run_002"], metric="best_val_loss")
"""
from __future__ import annotations

import logging

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.rade_ml_pt.tracking.run import Run

logger = logging.getLogger(__name__)

_RUN_FILENAME = "run.json"


class ExperimentTracker:
    """
    Local filesystem experiment tracker.

    Parameters
    ----------
    store_dir : str or Path
        Root directory where run records are persisted.
    """

    def __init__(self, store_dir: Union[str, Path]) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        name: str = "",
        tags: Optional[List[str]] = None,
    ) -> Run:
        """
        Start a new experiment run.

        Parameters
        ----------
        name : str
            Human-readable name.
        tags : list of str, optional
            Labels for later filtering.

        Returns
        -------
        Run
            A mutable run object.  Call ``run.end()`` when finished; the
            tracker will persist it via :meth:`save_run`.
        """
        run = Run(name=name, tags=tags or [])
        self._persist(run)
        logger.info(f"Started run '{run.run_id}' ({name})")
        return run

    def save_run(self, run: Run) -> None:
        """Persist the current state of a run to disk."""
        self._persist(run)

    def end_run(self, run: Run) -> None:
        """Mark a run as completed and persist it."""
        run.end()
        self._persist(run)
        logger.info(f"Ended run '{run.run_id}'")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> Run:
        """
        Retrieve a single run by its ID.

        Raises
        ------
        FileNotFoundError
            If the run does not exist on disk.
        """
        run_path = self.store_dir / run_id / _RUN_FILENAME
        if not run_path.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found in {self.store_dir}")
        return Run.from_json(run_path)

    def list_runs(
        self,
        tag: Optional[str] = None,
        sort_by: Optional[str] = None,
        ascending: bool = True,
    ) -> List[Run]:
        """
        List all recorded runs, optionally filtered by tag and sorted by a metric.

        Parameters
        ----------
        tag : str, optional
            Only include runs that carry this tag.
        sort_by : str, optional
            Metric key to sort by (e.g. "best_val_loss").
        ascending : bool
            Sort direction when ``sort_by`` is provided.

        Returns
        -------
        list of Run
        """
        runs: List[Run] = []
        for run_dir in sorted(self.store_dir.iterdir()):
            run_path = run_dir / _RUN_FILENAME
            if not run_path.exists():
                continue
            run = Run.from_json(run_path)
            if tag is None or tag in run.tags:
                runs.append(run)

        if sort_by is not None:
            runs.sort(
                key=lambda r: r.metrics.get(sort_by, float("inf")),
                reverse=not ascending,
            )
        return runs

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build a comparison DataFrame for a set of runs.

        Parameters
        ----------
        run_ids : list of str
            IDs of the runs to compare.
        metrics : list of str, optional
            Metric keys to include.  Defaults to the union of all metrics
            across the selected runs.

        Returns
        -------
        pd.DataFrame
            Rows = runs, columns = metrics.
        """
        runs = [self.get_run(rid) for rid in run_ids]

        if metrics is None:
            all_keys: set = set()
            for r in runs:
                all_keys.update(r.metrics.keys())
            metrics = sorted(all_keys)

        rows = []
        for r in runs:
            row: Dict[str, Any] = {"run_id": r.run_id, "name": r.name, "status": r.status}
            for m in metrics:
                row[m] = r.metrics.get(m)
            rows.append(row)

        return pd.DataFrame(rows)

    def delete_run(self, run_id: str) -> None:
        """Remove a run directory from the store."""
        import shutil

        run_dir = self.store_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found in {self.store_dir}")
        shutil.rmtree(run_dir)
        logger.info(f"Deleted run '{run_id}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist(self, run: Run) -> None:
        """Write a run's JSON to its run directory."""
        run_dir = self.store_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run.to_json(run_dir / _RUN_FILENAME)
