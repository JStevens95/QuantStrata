"""Unit tests for rade_ml.tracking.tracker -- ExperimentTracker."""
import pytest
import pandas as pd

from src.rade_ml.tracking.tracker import ExperimentTracker
from src.rade_ml.tracking.run import Run


class TestExperimentTrackerLifecycle:
    def test_start_run(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = tracker.start_run(name="test", tags=["tag1"])
        assert isinstance(run, Run)
        assert run.name == "test"
        assert run.status == "running"

    def test_end_run(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = tracker.start_run(name="test")
        run.log_metric("loss", 0.1)
        tracker.end_run(run)
        assert run.status == "completed"
        assert run.end_time is not None

    def test_save_and_get_run(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = tracker.start_run(name="test")
        run.log_metric("loss", 0.5)
        tracker.save_run(run)

        loaded = tracker.get_run(run.run_id)
        assert loaded.metrics["loss"] == 0.5

    def test_get_nonexistent_raises(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        with pytest.raises(FileNotFoundError):
            tracker.get_run("nonexistent_id")


class TestExperimentTrackerQueries:
    def test_list_runs(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        tracker.start_run(name="a", tags=["t1"])
        tracker.start_run(name="b", tags=["t2"])
        runs = tracker.list_runs()
        assert len(runs) == 2

    def test_list_runs_by_tag(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        tracker.start_run(name="a", tags=["sweep"])
        tracker.start_run(name="b", tags=["other"])
        runs = tracker.list_runs(tag="sweep")
        assert len(runs) == 1
        assert runs[0].name == "a"

    def test_list_runs_sorted(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        r1 = tracker.start_run(name="high_loss")
        r1.log_metric("loss", 0.9)
        tracker.save_run(r1)

        r2 = tracker.start_run(name="low_loss")
        r2.log_metric("loss", 0.1)
        tracker.save_run(r2)

        runs = tracker.list_runs(sort_by="loss", ascending=True)
        assert runs[0].metrics["loss"] < runs[1].metrics["loss"]

    def test_compare_runs(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        r1 = tracker.start_run(name="a")
        r1.log_metrics({"loss": 0.5, "mae": 0.3})
        tracker.save_run(r1)

        r2 = tracker.start_run(name="b")
        r2.log_metrics({"loss": 0.2, "mae": 0.1})
        tracker.save_run(r2)

        df = tracker.compare_runs([r1.run_id, r2.run_id])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "loss" in df.columns


class TestExperimentTrackerDelete:
    def test_delete_run(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        run = tracker.start_run(name="deleteme")
        run_id = run.run_id
        tracker.delete_run(run_id)
        with pytest.raises(FileNotFoundError):
            tracker.get_run(run_id)

    def test_delete_nonexistent_raises(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "runs")
        with pytest.raises(FileNotFoundError):
            tracker.delete_run("nonexistent")
