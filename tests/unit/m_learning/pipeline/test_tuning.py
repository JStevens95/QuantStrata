"""Tests for m_learning.pipeline.tuning."""

import pytest

from src.m_learning.pipeline.tuning import run_tuning, _grid_search
from src.m_learning.core.types import TuningResult


class TestGridSearch:
    """Tests for _grid_search helper."""

    def test_single_key(self):
        """Single parameter grid."""
        space = {"a": [1, 2, 3]}
        trials = _grid_search(lambda c: 0.0, space)
        assert len(trials) == 3
        assert [t["a"] for t in trials] == [1, 2, 3]

    def test_two_keys(self):
        """Two parameter grid (Cartesian product)."""
        space = {"a": [1, 2], "b": [10, 20]}
        trials = _grid_search(lambda c: 0.0, space)
        assert len(trials) == 4
        configs = [(t["a"], t["b"]) for t in trials]
        assert set(configs) == {(1, 10), (1, 20), (2, 10), (2, 20)}


class TestRunTuning:
    """Tests for run_tuning."""

    def test_grid_returns_tuning_result(self):
        """run_tuning with method=grid returns TuningResult."""
        def objective(config):
            return config["x"] ** 2  # minimize x^2 -> best x=0

        result = run_tuning(
            objective,
            search_space={"x": [0.0, 1.0, 2.0]},
            method="grid",
            minimize=True,
        )
        assert isinstance(result, TuningResult)
        assert result.best_config["x"] == 0.0
        assert result.best_score == 0.0
        assert len(result.trials) == 3
        assert result.metadata["method"] == "grid"
        assert result.metadata["n_trials"] == 3

    def test_grid_maximize(self):
        """run_tuning with minimize=False selects largest score."""
        def objective(config):
            return config["x"]

        result = run_tuning(
            objective,
            search_space={"x": [1.0, 2.0, 3.0]},
            method="grid",
            minimize=False,
        )
        assert result.best_config["x"] == 3.0
        assert result.best_score == 3.0

    def test_random_requires_n_trials(self):
        """run_tuning with method=random requires n_trials."""
        with pytest.raises(ValueError, match="n_trials"):
            run_tuning(
                lambda c: 0.0,
                search_space={"x": [1, 2, 3]},
                method="random",
            )

    def test_random_respects_n_trials(self):
        """run_tuning with method=random runs n_trials."""
        def objective(config):
            return config["x"]

        result = run_tuning(
            objective,
            search_space={"x": [1, 2, 3]},
            method="random",
            n_trials=5,
            minimize=True,
            metadata={"seed": 42},
        )
        assert isinstance(result, TuningResult)
        assert len(result.trials) == 5
        assert result.metadata["n_trials"] == 5
