"""
Experiment tracking protocols and implementations for ML pipelines.

Provides a unified interface for experiment tracking with support for:
- MLflow tracking
- Weights & Biases (W&B) tracking
- In-memory tracking (for testing/local development)

Usage:
    from src.machine_learning.core.tracking import (
        create_tracker, MLflowTracker, WandBTracker, InMemoryTracker
    )
    
    # Create tracker based on available backend
    tracker = create_tracker("mlflow", experiment_name="my_experiment")
    
    # Start a run
    with tracker.start_run(run_name="training_v1") as run:
        # Log parameters
        tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
        
        # Log metrics during training
        for epoch in range(100):
            tracker.log_metrics({"loss": loss, "accuracy": acc}, step=epoch)
        
        # Log artifacts
        tracker.log_artifact("model.h5", artifact_path="models")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Union, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols and Types
# =============================================================================


@dataclass
class RunInfo:
    """Information about a tracking run."""
    
    run_id: str
    run_name: Optional[str]
    experiment_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "RUNNING"
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


@runtime_checkable
class ExperimentTracker(Protocol):
    """
    Protocol for experiment tracking backends.
    
    Any tracking implementation must provide these methods for integration
    with the ML training pipelines.
    """
    
    @property
    def experiment_name(self) -> str:
        """Return the current experiment name."""
        ...
    
    @property
    def current_run(self) -> Optional[RunInfo]:
        """Return information about the current active run, or None."""
        ...
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> "ExperimentTracker":
        """
        Start a new tracking run.
        
        Parameters
        ----------
        run_name : str, optional
            Human-readable name for the run.
        tags : dict, optional
            Tags to associate with the run.
            
        Returns
        -------
        ExperimentTracker
            Self, for context manager usage.
        """
        ...
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.
        
        Parameters
        ----------
        status : str
            Final status of the run (FINISHED, FAILED, etc.).
        """
        ...
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Parameters
        ----------
        params : dict
            Dictionary of parameter names to values.
        """
        ...
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of metric names to values.
        step : int, optional
            Step/epoch number for the metrics.
        """
        ...
    
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log an artifact (file or directory).
        
        Parameters
        ----------
        local_path : str or Path
            Path to the local file or directory.
        artifact_path : str, optional
            Destination path within the artifact store.
        """
        ...
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags on the current run.
        
        Parameters
        ----------
        tags : dict
            Dictionary of tag names to values.
        """
        ...


# =============================================================================
# In-Memory Tracker (for testing and local development)
# =============================================================================


class InMemoryTracker:
    """
    In-memory experiment tracker for testing and local development.
    
    Stores all tracking data in memory without external dependencies.
    Useful for unit tests and environments without MLflow/W&B.
    
    Example:
        tracker = InMemoryTracker(experiment_name="test_experiment")
        with tracker.start_run("run_1"):
            tracker.log_params({"lr": 0.001})
            tracker.log_metrics({"loss": 0.5}, step=1)
        
        # Access stored data
        runs = tracker.get_all_runs()
    """
    
    def __init__(
        self,
        experiment_name: str = "default",
        artifact_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize in-memory tracker.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        artifact_dir : str or Path, optional
            Directory to store artifacts. If None, artifacts are not persisted.
        """
        self._experiment_name = experiment_name
        self._artifact_dir = Path(artifact_dir) if artifact_dir else None
        self._runs: Dict[str, RunInfo] = {}
        self._current_run: Optional[RunInfo] = None
        self._run_counter = 0
    
    @property
    def experiment_name(self) -> str:
        """Return the experiment name."""
        return self._experiment_name
    
    @property
    def current_run(self) -> Optional[RunInfo]:
        """Return the current active run."""
        return self._current_run
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Iterator["InMemoryTracker"]:
        """Start a new tracking run as a context manager."""
        self._run_counter += 1
        run_id = f"run_{self._run_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._current_run = RunInfo(
            run_id=run_id,
            run_name=run_name or f"run_{self._run_counter}",
            experiment_name=self._experiment_name,
            start_time=datetime.now(),
            tags=tags or {},
        )
        self._runs[run_id] = self._current_run
        
        logger.info("Started run %s (name=%s)", run_id, run_name)
        
        try:
            yield self
        except Exception as e:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        if self._current_run is not None:
            self._current_run.end_time = datetime.now()
            self._current_run.status = status
            logger.info(
                "Ended run %s with status %s",
                self._current_run.run_id,
                status,
            )
            self._current_run = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to the current run."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        self._current_run.params.update(params)
        logger.debug("Logged params: %s", params)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to the current run."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        for name, value in metrics.items():
            if name not in self._current_run.metrics:
                self._current_run.metrics[name] = []
            self._current_run.metrics[name].append({
                "value": value,
                "step": step,
                "timestamp": datetime.now().isoformat(),
            })
        
        logger.debug("Logged metrics at step %s: %s", step, metrics)
    
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log an artifact to the current run."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        local_path = Path(local_path)
        
        if self._artifact_dir is not None:
            # Copy artifact to artifact directory
            run_artifact_dir = (
                self._artifact_dir / self._current_run.run_id / (artifact_path or "")
            )
            run_artifact_dir.mkdir(parents=True, exist_ok=True)
            
            if local_path.is_file():
                dest = run_artifact_dir / local_path.name
                shutil.copy2(local_path, dest)
            elif local_path.is_dir():
                dest = run_artifact_dir / local_path.name
                shutil.copytree(local_path, dest, dirs_exist_ok=True)
        
        self._current_run.artifacts.append(str(local_path))
        logger.debug("Logged artifact: %s", local_path)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on the current run."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        self._current_run.tags.update(tags)
        logger.debug("Set tags: %s", tags)
    
    def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Get a specific run by ID."""
        return self._runs.get(run_id)
    
    def get_all_runs(self) -> List[RunInfo]:
        """Get all runs in the experiment."""
        return list(self._runs.values())
    
    def get_best_run(
        self,
        metric: str,
        minimize: bool = True,
    ) -> Optional[RunInfo]:
        """
        Get the run with the best value for a metric.
        
        Parameters
        ----------
        metric : str
            Name of the metric to compare.
        minimize : bool
            If True, return run with lowest metric; else highest.
            
        Returns
        -------
        RunInfo or None
            The best run, or None if no runs have the metric.
        """
        best_run = None
        best_value = float("inf") if minimize else float("-inf")
        
        for run in self._runs.values():
            if metric in run.metrics and run.metrics[metric]:
                # Get the last logged value for the metric
                last_value = run.metrics[metric][-1]["value"]
                if minimize and last_value < best_value:
                    best_value = last_value
                    best_run = run
                elif not minimize and last_value > best_value:
                    best_value = last_value
                    best_run = run
        
        return best_run


# =============================================================================
# MLflow Tracker
# =============================================================================


class MLflowTracker:
    """
    MLflow experiment tracker.
    
    Wraps MLflow tracking API for experiment tracking, parameter logging,
    metric logging, and artifact storage.
    
    Requires: pip install mlflow
    
    Example:
        tracker = MLflowTracker(
            experiment_name="option_pricer",
            tracking_uri="http://localhost:5000",
        )
        
        with tracker.start_run("training_v1"):
            tracker.log_params({"learning_rate": 0.001})
            for epoch in range(100):
                tracker.log_metrics({"loss": loss}, step=epoch)
            tracker.log_artifact("model.h5")
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ) -> None:
        """
        Initialize MLflow tracker.
        
        Parameters
        ----------
        experiment_name : str
            Name of the MLflow experiment.
        tracking_uri : str, optional
            MLflow tracking server URI. If None, uses local filesystem.
        artifact_location : str, optional
            Location for artifact storage.
        """
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError as e:
            raise ImportError(
                "MLflow is required for MLflowTracker. "
                "Install with: pip install mlflow"
            ) from e
        
        self._experiment_name = experiment_name
        self._current_run: Optional[RunInfo] = None
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        else:
            self._experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
    
    @property
    def experiment_name(self) -> str:
        """Return the experiment name."""
        return self._experiment_name
    
    @property
    def current_run(self) -> Optional[RunInfo]:
        """Return the current active run."""
        return self._current_run
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Iterator["MLflowTracker"]:
        """Start a new MLflow run as a context manager."""
        with self._mlflow.start_run(run_name=run_name, tags=tags) as run:
            self._current_run = RunInfo(
                run_id=run.info.run_id,
                run_name=run_name,
                experiment_name=self._experiment_name,
                start_time=datetime.now(),
                tags=tags or {},
            )
            
            logger.info("Started MLflow run %s", run.info.run_id)
            
            try:
                yield self
            except Exception as e:
                self._mlflow.set_tag("status", "FAILED")
                self._current_run.status = "FAILED"
                raise
            finally:
                self._current_run.end_time = datetime.now()
                self._current_run = None
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run (typically handled by context manager)."""
        if self._current_run is not None:
            self._mlflow.set_tag("status", status)
            self._mlflow.end_run()
            self._current_run = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        # MLflow requires string values, so convert
        str_params = {k: str(v) for k, v in params.items()}
        self._mlflow.log_params(str_params)
        
        if self._current_run:
            self._current_run.params.update(params)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to MLflow."""
        self._mlflow.log_metrics(metrics, step=step)
        
        if self._current_run:
            for name, value in metrics.items():
                if name not in self._current_run.metrics:
                    self._current_run.metrics[name] = []
                self._current_run.metrics[name].append({
                    "value": value,
                    "step": step,
                })
    
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log an artifact to MLflow."""
        self._mlflow.log_artifact(str(local_path), artifact_path)
        
        if self._current_run:
            self._current_run.artifacts.append(str(local_path))
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on the current run."""
        self._mlflow.set_tags(tags)
        
        if self._current_run:
            self._current_run.tags.update(tags)


# =============================================================================
# Weights & Biases Tracker
# =============================================================================


class WandBTracker:
    """
    Weights & Biases experiment tracker.
    
    Wraps W&B API for experiment tracking with rich visualizations
    and collaboration features.
    
    Requires: pip install wandb
    
    Example:
        tracker = WandBTracker(
            project="option_pricer",
            entity="my_team",
        )
        
        with tracker.start_run("training_v1", config={"lr": 0.001}):
            for epoch in range(100):
                tracker.log_metrics({"loss": loss}, step=epoch)
    """
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
    ) -> None:
        """
        Initialize W&B tracker.
        
        Parameters
        ----------
        project : str
            W&B project name.
        entity : str, optional
            W&B entity (team or username).
        group : str, optional
            Group name for organizing runs.
        job_type : str, optional
            Type of job (e.g., "training", "evaluation").
        """
        try:
            import wandb
            self._wandb = wandb
        except ImportError as e:
            raise ImportError(
                "Weights & Biases is required for WandBTracker. "
                "Install with: pip install wandb"
            ) from e
        
        self._project = project
        self._entity = entity
        self._group = group
        self._job_type = job_type
        self._current_run: Optional[RunInfo] = None
        self._wandb_run = None
    
    @property
    def experiment_name(self) -> str:
        """Return the project name as experiment name."""
        return self._project
    
    @property
    def current_run(self) -> Optional[RunInfo]:
        """Return the current active run."""
        return self._current_run
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Iterator["WandBTracker"]:
        """Start a new W&B run as a context manager."""
        self._wandb_run = self._wandb.init(
            project=self._project,
            entity=self._entity,
            group=self._group,
            job_type=self._job_type,
            name=run_name,
            tags=list(tags.values()) if tags else None,
            config=config,
        )
        
        self._current_run = RunInfo(
            run_id=self._wandb_run.id,
            run_name=run_name,
            experiment_name=self._project,
            start_time=datetime.now(),
            tags=tags or {},
            params=config or {},
        )
        
        logger.info("Started W&B run %s", self._wandb_run.id)
        
        try:
            yield self
        except Exception as e:
            self._current_run.status = "FAILED"
            raise
        finally:
            self._wandb.finish()
            self._current_run.end_time = datetime.now()
            self._current_run = None
            self._wandb_run = None
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        if self._wandb_run is not None:
            self._wandb.finish()
            self._wandb_run = None
            self._current_run = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to W&B config."""
        if self._wandb_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        self._wandb_run.config.update(params)
        
        if self._current_run:
            self._current_run.params.update(params)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to W&B."""
        if self._wandb_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        log_dict = {**metrics}
        if step is not None:
            log_dict["step"] = step
        
        self._wandb.log(log_dict, step=step)
        
        if self._current_run:
            for name, value in metrics.items():
                if name not in self._current_run.metrics:
                    self._current_run.metrics[name] = []
                self._current_run.metrics[name].append({
                    "value": value,
                    "step": step,
                })
    
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
        artifact_type: str = "model",
    ) -> None:
        """Log an artifact to W&B."""
        if self._wandb_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        artifact_name = artifact_path or Path(local_path).stem
        artifact = self._wandb.Artifact(artifact_name, type=artifact_type)
        
        local_path = Path(local_path)
        if local_path.is_file():
            artifact.add_file(str(local_path))
        elif local_path.is_dir():
            artifact.add_dir(str(local_path))
        
        self._wandb_run.log_artifact(artifact)
        
        if self._current_run:
            self._current_run.artifacts.append(str(local_path))
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on the current run."""
        if self._wandb_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # W&B uses a list of tags
        existing_tags = list(self._wandb_run.tags) if self._wandb_run.tags else []
        new_tags = list(set(existing_tags + list(tags.values())))
        self._wandb_run.tags = tuple(new_tags)
        
        if self._current_run:
            self._current_run.tags.update(tags)


# =============================================================================
# Factory Function
# =============================================================================


def create_tracker(
    backend: str = "memory",
    experiment_name: str = "default",
    **kwargs: Any,
) -> ExperimentTracker:
    """
    Create an experiment tracker for the specified backend.
    
    Parameters
    ----------
    backend : str
        Tracking backend: "memory", "mlflow", or "wandb".
    experiment_name : str
        Name of the experiment (or project for W&B).
    **kwargs
        Additional arguments passed to the tracker constructor.
        
        For MLflow:
            - tracking_uri: MLflow server URI
            - artifact_location: Artifact storage location
            
        For W&B:
            - entity: W&B entity (team/user)
            - group: Run group
            - job_type: Job type
            
        For memory:
            - artifact_dir: Directory for artifact storage
    
    Returns
    -------
    ExperimentTracker
        The configured tracker instance.
    
    Example:
        # In-memory (default)
        tracker = create_tracker()
        
        # MLflow
        tracker = create_tracker(
            "mlflow",
            experiment_name="my_experiment",
            tracking_uri="http://localhost:5000",
        )
        
        # W&B
        tracker = create_tracker(
            "wandb",
            experiment_name="my_project",
            entity="my_team",
        )
    """
    backend = backend.lower()
    
    if backend in ("memory", "inmemory", "in_memory"):
        return InMemoryTracker(
            experiment_name=experiment_name,
            artifact_dir=kwargs.get("artifact_dir"),
        )
    
    elif backend == "mlflow":
        return MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri=kwargs.get("tracking_uri"),
            artifact_location=kwargs.get("artifact_location"),
        )
    
    elif backend in ("wandb", "wb", "weights_and_biases"):
        return WandBTracker(
            project=experiment_name,
            entity=kwargs.get("entity"),
            group=kwargs.get("group"),
            job_type=kwargs.get("job_type"),
        )
    
    else:
        raise ValueError(
            f"Unknown tracking backend: {backend!r}. "
            f"Choose from: 'memory', 'mlflow', 'wandb'"
        )


__all__ = [
    # Protocols and types
    "ExperimentTracker",
    "RunInfo",
    # Implementations
    "InMemoryTracker",
    "MLflowTracker",
    "WandBTracker",
    # Factory
    "create_tracker",
]
