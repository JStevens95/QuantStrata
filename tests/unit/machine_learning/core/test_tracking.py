"""
Unit tests for experiment tracking module.

Tests InMemoryTracker, MLflowTracker, WandBTracker, and create_tracker factory.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Skip entire module if TensorFlow is not available (core imports base which needs TF)
pytest.importorskip("tensorflow")

from src.machine_learning.core.tracking import (
    InMemoryTracker,
    RunInfo,
    create_tracker,
)


class TestRunInfo:
    """Tests for RunInfo dataclass."""
    
    def test_run_info_creation(self) -> None:
        """Test basic RunInfo creation."""
        run = RunInfo(
            run_id="test_run_1",
            run_name="test",
            experiment_name="experiment",
            start_time=datetime.now(),
        )
        
        assert run.run_id == "test_run_1"
        assert run.run_name == "test"
        assert run.status == "RUNNING"
        assert run.params == {}
        assert run.metrics == {}
    
    def test_run_info_with_metrics(self) -> None:
        """Test RunInfo with metrics."""
        run = RunInfo(
            run_id="test_run_2",
            run_name="test",
            experiment_name="experiment",
            start_time=datetime.now(),
            metrics={"loss": [{"value": 0.5, "step": 1}]},
        )
        
        assert "loss" in run.metrics
        assert run.metrics["loss"][0]["value"] == 0.5


class TestInMemoryTracker:
    """Tests for InMemoryTracker."""
    
    def test_tracker_creation(self) -> None:
        """Test tracker initialisation."""
        tracker = InMemoryTracker(experiment_name="test_experiment")
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.current_run is None
    
    def test_start_run_context_manager(self) -> None:
        """Test run management with context manager."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with tracker.start_run(run_name="run_1") as run_tracker:
            assert tracker.current_run is not None
            assert tracker.current_run.run_name == "run_1"
            assert run_tracker is tracker
        
        # Run should be finished after context
        assert tracker.current_run is None
    
    def test_log_params(self) -> None:
        """Test parameter logging."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with tracker.start_run("run_1"):
            tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
            
            assert tracker.current_run.params["learning_rate"] == 0.001
            assert tracker.current_run.params["batch_size"] == 32
    
    def test_log_params_no_run_raises(self) -> None:
        """Test that logging params without active run raises error."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_params({"lr": 0.001})
    
    def test_log_metrics(self) -> None:
        """Test metric logging."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with tracker.start_run("run_1"):
            tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
            tracker.log_metrics({"loss": 0.3, "accuracy": 0.95}, step=2)
            
            assert len(tracker.current_run.metrics["loss"]) == 2
            assert tracker.current_run.metrics["loss"][0]["value"] == 0.5
            assert tracker.current_run.metrics["loss"][1]["value"] == 0.3
    
    def test_log_metrics_no_run_raises(self) -> None:
        """Test that logging metrics without active run raises error."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_metrics({"loss": 0.5})
    
    def test_set_tags(self) -> None:
        """Test tag setting."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with tracker.start_run("run_1"):
            tracker.set_tags({"model_type": "gnn", "asset_class": "fx"})
            
            assert tracker.current_run.tags["model_type"] == "gnn"
            assert tracker.current_run.tags["asset_class"] == "fx"
    
    def test_log_artifact(self) -> None:
        """Test artifact logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            tracker = InMemoryTracker(
                experiment_name="test",
                artifact_dir=artifact_dir,
            )
            
            # Create a test file
            test_file = Path(tmpdir) / "model.txt"
            test_file.write_text("test model")
            
            with tracker.start_run("run_1"):
                tracker.log_artifact(test_file)
                
                assert str(test_file) in tracker.current_run.artifacts
    
    def test_get_all_runs(self) -> None:
        """Test retrieving all runs."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with tracker.start_run("run_1"):
            tracker.log_metrics({"loss": 0.5})
        
        with tracker.start_run("run_2"):
            tracker.log_metrics({"loss": 0.3})
        
        runs = tracker.get_all_runs()
        assert len(runs) == 2
    
    def test_get_best_run_minimize(self) -> None:
        """Test getting best run when minimizing."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with tracker.start_run("run_1"):
            tracker.log_metrics({"loss": 0.5})
        
        with tracker.start_run("run_2"):
            tracker.log_metrics({"loss": 0.3})
        
        best = tracker.get_best_run("loss", minimize=True)
        assert best is not None
        assert best.metrics["loss"][-1]["value"] == 0.3
    
    def test_get_best_run_maximize(self) -> None:
        """Test getting best run when maximizing."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with tracker.start_run("run_1"):
            tracker.log_metrics({"accuracy": 0.9})
        
        with tracker.start_run("run_2"):
            tracker.log_metrics({"accuracy": 0.95})
        
        best = tracker.get_best_run("accuracy", minimize=False)
        assert best is not None
        assert best.metrics["accuracy"][-1]["value"] == 0.95
    
    def test_run_status_on_exception(self) -> None:
        """Test that run status is FAILED on exception."""
        tracker = InMemoryTracker(experiment_name="test")
        
        with pytest.raises(ValueError):
            with tracker.start_run("run_1"):
                raise ValueError("Test error")
        
        runs = tracker.get_all_runs()
        assert runs[0].status == "FAILED"


class TestCreateTracker:
    """Tests for create_tracker factory function."""
    
    def test_create_memory_tracker(self) -> None:
        """Test creating in-memory tracker."""
        tracker = create_tracker("memory", experiment_name="test")
        
        assert isinstance(tracker, InMemoryTracker)
        assert tracker.experiment_name == "test"
    
    def test_create_inmemory_tracker_alias(self) -> None:
        """Test inmemory alias."""
        tracker = create_tracker("inmemory", experiment_name="test")
        
        assert isinstance(tracker, InMemoryTracker)
    
    def test_create_unknown_backend_raises(self) -> None:
        """Test that unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown tracking backend"):
            create_tracker("unknown_backend")
    
    def test_default_experiment_name(self) -> None:
        """Test default experiment name."""
        tracker = create_tracker()
        
        assert tracker.experiment_name == "default"


class TestTrackerProtocol:
    """Tests for ExperimentTracker protocol compliance."""
    
    def test_inmemory_implements_protocol(self) -> None:
        """Test that InMemoryTracker implements the protocol."""
        from src.machine_learning.core.tracking import ExperimentTracker
        
        tracker = InMemoryTracker(experiment_name="test")
        
        # Check protocol compliance
        assert isinstance(tracker, ExperimentTracker)
        assert hasattr(tracker, "experiment_name")
        assert hasattr(tracker, "current_run")
        assert hasattr(tracker, "start_run")
        assert hasattr(tracker, "end_run")
        assert hasattr(tracker, "log_params")
        assert hasattr(tracker, "log_metrics")
        assert hasattr(tracker, "log_artifact")
        assert hasattr(tracker, "set_tags")
