"""
Unit tests for experiment tracking module.

Tests the InMemoryTracker, MLflowTracker, WandBTracker, and factory function.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Import directly from module to avoid triggering full machine_learning import
sys.path.insert(0, str(Path(__file__).parents[4]))
from src.machine_learning.core.tracking import (
    InMemoryTracker,
    RunInfo,
    create_tracker,
)


class TestRunInfo:
    """Tests for RunInfo dataclass."""
    
    def test_run_info_creation(self):
        """Test creating RunInfo."""
        run = RunInfo(
            run_id="test_run_1",
            run_name="Test Run",
            experiment_name="test_experiment",
            start_time=datetime.now(),
        )
        
        assert run.run_id == "test_run_1"
        assert run.run_name == "Test Run"
        assert run.experiment_name == "test_experiment"
        assert run.status == "RUNNING"
        assert run.params == {}
        assert run.metrics == {}
        assert run.tags == {}
        assert run.artifacts == []
    
    def test_run_info_with_data(self):
        """Test RunInfo with data populated."""
        run = RunInfo(
            run_id="test_run_2",
            run_name="Test Run 2",
            experiment_name="test_experiment",
            start_time=datetime.now(),
            params={"lr": 0.001},
            metrics={"loss": [{"value": 0.5, "step": 1}]},
            tags={"version": "1.0"},
        )
        
        assert run.params == {"lr": 0.001}
        assert len(run.metrics["loss"]) == 1


class TestInMemoryTracker:
    """Tests for InMemoryTracker."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        assert tracker.experiment_name == "test_exp"
        assert tracker.current_run is None
        assert len(tracker.get_all_runs()) == 0
    
    def test_start_and_end_run(self):
        """Test starting and ending a run."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        with tracker.start_run(run_name="test_run") as t:
            assert t.current_run is not None
            assert t.current_run.run_name == "test_run"
            assert t.current_run.status == "RUNNING"
        
        # After context manager exits
        assert tracker.current_run is None
        runs = tracker.get_all_runs()
        assert len(runs) == 1
        assert runs[0].status == "FINISHED"
    
    def test_log_params(self):
        """Test logging parameters."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        with tracker.start_run("test_run"):
            tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
            tracker.log_params({"epochs": 100})
        
        run = tracker.get_all_runs()[0]
        assert run.params["learning_rate"] == 0.001
        assert run.params["batch_size"] == 32
        assert run.params["epochs"] == 100
    
    def test_log_metrics(self):
        """Test logging metrics."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        with tracker.start_run("test_run"):
            tracker.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
            tracker.log_metrics({"loss": 0.3, "accuracy": 0.9}, step=2)
        
        run = tracker.get_all_runs()[0]
        assert len(run.metrics["loss"]) == 2
        assert run.metrics["loss"][0]["value"] == 0.5
        assert run.metrics["loss"][0]["step"] == 1
        assert run.metrics["loss"][1]["value"] == 0.3
    
    def test_log_artifact_with_file(self):
        """Test logging file artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temp file
            test_file = Path(tmpdir) / "model.txt"
            test_file.write_text("test model")
            
            artifact_dir = Path(tmpdir) / "artifacts"
            tracker = InMemoryTracker(
                experiment_name="test_exp",
                artifact_dir=artifact_dir,
            )
            
            with tracker.start_run("test_run") as t:
                tracker.log_artifact(test_file)
                run_id = t.current_run.run_id
            
            # Check artifact was copied
            run = tracker.get_all_runs()[0]
            assert str(test_file) in run.artifacts
            
            # Check file was copied to artifact directory
            copied_file = artifact_dir / run_id / "model.txt"
            assert copied_file.exists()
            assert copied_file.read_text() == "test model"
    
    def test_set_tags(self):
        """Test setting tags."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        with tracker.start_run("test_run"):
            tracker.set_tags({"version": "1.0", "env": "test"})
        
        run = tracker.get_all_runs()[0]
        assert run.tags["version"] == "1.0"
        assert run.tags["env"] == "test"
    
    def test_multiple_runs(self):
        """Test multiple runs in same experiment."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        for i in range(3):
            with tracker.start_run(f"run_{i}"):
                tracker.log_metrics({"score": float(i)})
        
        runs = tracker.get_all_runs()
        assert len(runs) == 3
        assert all(r.status == "FINISHED" for r in runs)
    
    def test_get_run(self):
        """Test getting a specific run."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        with tracker.start_run("test_run") as t:
            run_id = t.current_run.run_id
        
        run = tracker.get_run(run_id)
        assert run is not None
        assert run.run_name == "test_run"
        
        assert tracker.get_run("nonexistent") is None
    
    def test_get_best_run_minimize(self):
        """Test getting best run by minimizing metric."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        for i, loss in enumerate([0.5, 0.2, 0.8]):
            with tracker.start_run(f"run_{i}"):
                tracker.log_metrics({"loss": loss})
        
        best = tracker.get_best_run("loss", minimize=True)
        assert best is not None
        assert best.metrics["loss"][-1]["value"] == 0.2
    
    def test_get_best_run_maximize(self):
        """Test getting best run by maximizing metric."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        for i, acc in enumerate([0.8, 0.95, 0.7]):
            with tracker.start_run(f"run_{i}"):
                tracker.log_metrics({"accuracy": acc})
        
        best = tracker.get_best_run("accuracy", minimize=False)
        assert best is not None
        assert best.metrics["accuracy"][-1]["value"] == 0.95
    
    def test_error_without_active_run(self):
        """Test that operations fail without active run."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_params({"lr": 0.001})
        
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_metrics({"loss": 0.5})
    
    def test_failed_run(self):
        """Test that exceptions mark run as failed."""
        tracker = InMemoryTracker(experiment_name="test_exp")
        
        with pytest.raises(ValueError):
            with tracker.start_run("failing_run"):
                tracker.log_params({"lr": 0.001})
                raise ValueError("Test error")
        
        runs = tracker.get_all_runs()
        assert len(runs) == 1
        assert runs[0].status == "FAILED"


class TestCreateTracker:
    """Tests for create_tracker factory function."""
    
    def test_create_memory_tracker(self):
        """Test creating in-memory tracker."""
        tracker = create_tracker(
            backend="memory",
            experiment_name="test_exp",
        )
        
        assert isinstance(tracker, InMemoryTracker)
        assert tracker.experiment_name == "test_exp"
    
    def test_create_memory_tracker_with_artifact_dir(self):
        """Test creating in-memory tracker with artifact directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = create_tracker(
                backend="memory",
                experiment_name="test_exp",
                artifact_dir=tmpdir,
            )
            
            assert tracker._artifact_dir == Path(tmpdir)
    
    def test_create_tracker_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown tracking backend"):
            create_tracker(backend="invalid")
    
    def test_create_mlflow_tracker_import_error(self):
        """Test that MLflow tracker requires mlflow package."""
        # This test only runs if mlflow is not installed
        try:
            import mlflow
            pytest.skip("MLflow is installed")
        except ImportError:
            with pytest.raises(ImportError, match="MLflow is required"):
                create_tracker(backend="mlflow", experiment_name="test")
    
    def test_create_wandb_tracker_import_error(self):
        """Test that W&B tracker requires wandb package."""
        # This test only runs if wandb is not installed
        try:
            import wandb
            pytest.skip("W&B is installed")
        except ImportError:
            with pytest.raises(ImportError, match="Weights & Biases is required"):
                create_tracker(backend="wandb", experiment_name="test")


class TestExperimentTrackerProtocol:
    """Test that InMemoryTracker satisfies ExperimentTracker protocol."""
    
    def test_protocol_methods(self):
        """Test all protocol methods are present."""
        tracker = InMemoryTracker(experiment_name="test")
        
        # Check all required methods exist
        assert hasattr(tracker, "experiment_name")
        assert hasattr(tracker, "current_run")
        assert hasattr(tracker, "start_run")
        assert hasattr(tracker, "end_run")
        assert hasattr(tracker, "log_params")
        assert hasattr(tracker, "log_metrics")
        assert hasattr(tracker, "log_artifact")
        assert hasattr(tracker, "set_tags")
