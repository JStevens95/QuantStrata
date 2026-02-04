"""
Unit tests for hyperparameter tuning module.

Tests SearchSpace, parameter definitions, pruners, and tuning results.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.machine_learning.tuning.search_space import (
    MedianPruner,
    OptunaSearchSpace,
    ParameterDefinition,
    ParameterType,
    PercentilePruner,
    SearchSpace,
    TuningResult,
    TuningTrial,
    create_search_space,
)


class TestParameterDefinition:
    """Tests for ParameterDefinition."""
    
    def test_float_parameter(self) -> None:
        """Test float parameter definition."""
        param = ParameterDefinition(
            name="learning_rate",
            param_type=ParameterType.FLOAT,
            low=1e-4,
            high=1e-2,
            log=True,
        )
        
        assert param.name == "learning_rate"
        assert param.param_type == ParameterType.FLOAT
        assert param.low == 1e-4
        assert param.high == 1e-2
        assert param.log is True
    
    def test_int_parameter(self) -> None:
        """Test integer parameter definition."""
        param = ParameterDefinition(
            name="hidden_units",
            param_type=ParameterType.INT,
            low=32,
            high=256,
        )
        
        assert param.name == "hidden_units"
        assert param.param_type == ParameterType.INT
    
    def test_categorical_parameter(self) -> None:
        """Test categorical parameter definition."""
        param = ParameterDefinition(
            name="activation",
            param_type=ParameterType.CATEGORICAL,
            choices=["relu", "tanh", "gelu"],
        )
        
        assert param.name == "activation"
        assert param.choices == ["relu", "tanh", "gelu"]
    
    def test_float_parameter_missing_bounds_raises(self) -> None:
        """Test that float parameter without bounds raises."""
        with pytest.raises(ValueError, match="low and high required"):
            ParameterDefinition(
                name="lr",
                param_type=ParameterType.FLOAT,
            )
    
    def test_categorical_missing_choices_raises(self) -> None:
        """Test that categorical parameter without choices raises."""
        with pytest.raises(ValueError, match="choices required"):
            ParameterDefinition(
                name="opt",
                param_type=ParameterType.CATEGORICAL,
            )
    
    def test_sample_random_float(self) -> None:
        """Test random sampling for float parameter."""
        param = ParameterDefinition(
            name="lr",
            param_type=ParameterType.FLOAT,
            low=0.0,
            high=1.0,
        )
        
        for _ in range(10):
            value = param.sample_random()
            assert 0.0 <= value <= 1.0
    
    def test_sample_random_int(self) -> None:
        """Test random sampling for integer parameter."""
        param = ParameterDefinition(
            name="units",
            param_type=ParameterType.INT,
            low=10,
            high=100,
        )
        
        for _ in range(10):
            value = param.sample_random()
            assert 10 <= value <= 100
            assert isinstance(value, int)
    
    def test_sample_random_categorical(self) -> None:
        """Test random sampling for categorical parameter."""
        param = ParameterDefinition(
            name="opt",
            param_type=ParameterType.CATEGORICAL,
            choices=["adam", "sgd", "rmsprop"],
        )
        
        for _ in range(10):
            value = param.sample_random()
            assert value in ["adam", "sgd", "rmsprop"]


class TestSearchSpace:
    """Tests for SearchSpace."""
    
    def test_empty_search_space(self) -> None:
        """Test empty search space."""
        space = SearchSpace()
        
        assert len(space) == 0
        assert space.names == []
        assert space.parameters == []
    
    def test_add_float(self) -> None:
        """Test adding float parameter."""
        space = SearchSpace()
        result = space.add_float("lr", 1e-4, 1e-2, log=True)
        
        assert result is space  # Chaining
        assert "lr" in space
        assert len(space) == 1
    
    def test_add_int(self) -> None:
        """Test adding integer parameter."""
        space = SearchSpace()
        space.add_int("units", 32, 256)
        
        assert "units" in space
        param = space.get("units")
        assert param is not None
        assert param.param_type == ParameterType.INT
    
    def test_add_categorical(self) -> None:
        """Test adding categorical parameter."""
        space = SearchSpace()
        space.add_categorical("opt", ["adam", "sgd"])
        
        assert "opt" in space
        param = space.get("opt")
        assert param is not None
        assert param.choices == ["adam", "sgd"]
    
    def test_add_bool(self) -> None:
        """Test adding boolean parameter."""
        space = SearchSpace()
        space.add_bool("use_dropout")
        
        assert "use_dropout" in space
    
    def test_chaining(self) -> None:
        """Test fluent API chaining."""
        space = (
            SearchSpace()
            .add_float("lr", 1e-4, 1e-2, log=True)
            .add_int("units", 32, 256)
            .add_categorical("opt", ["adam", "sgd"])
            .add_bool("dropout")
        )
        
        assert len(space) == 4
    
    def test_sample(self) -> None:
        """Test sampling a configuration."""
        space = (
            SearchSpace()
            .add_float("lr", 0.0, 1.0)
            .add_int("units", 10, 100)
            .add_categorical("opt", ["adam", "sgd"])
        )
        
        config = space.sample()
        
        assert "lr" in config
        assert "units" in config
        assert "opt" in config
        assert 0.0 <= config["lr"] <= 1.0
        assert 10 <= config["units"] <= 100
        assert config["opt"] in ["adam", "sgd"]
    
    def test_sample_with_seed(self) -> None:
        """Test reproducible sampling with seed."""
        space = SearchSpace().add_float("lr", 0.0, 1.0)
        
        config1 = space.sample(seed=42)
        config2 = space.sample(seed=42)
        
        assert config1["lr"] == config2["lr"]
    
    def test_to_dict(self) -> None:
        """Test exporting to dictionary."""
        space = (
            SearchSpace()
            .add_float("lr", 1e-4, 1e-2, log=True)
            .add_int("units", 32, 256)
        )
        
        d = space.to_dict()
        
        assert "lr" in d
        assert d["lr"]["type"] == "float"
        assert d["lr"]["log"] is True
        assert "units" in d
        assert d["units"]["type"] == "int"
    
    def test_from_dict(self) -> None:
        """Test importing from dictionary."""
        config = {
            "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "units": {"type": "int", "low": 32, "high": 256},
            "opt": {"type": "categorical", "choices": ["adam", "sgd"]},
        }
        
        space = SearchSpace.from_dict(config)
        
        assert len(space) == 3
        assert space.get("lr").log is True


class TestOptunaSearchSpace:
    """Tests for OptunaSearchSpace wrapper."""
    
    def test_wrapper_creation(self) -> None:
        """Test creating wrapper."""
        space = SearchSpace().add_float("lr", 1e-4, 1e-2)
        optuna_space = OptunaSearchSpace(space)
        
        assert optuna_space._space is space


class TestMedianPruner:
    """Tests for MedianPruner."""
    
    def test_pruner_creation(self) -> None:
        """Test pruner creation with defaults."""
        pruner = MedianPruner()
        
        assert pruner.n_startup_trials == 5
        assert pruner.n_warmup_steps == 0
        assert pruner.interval_steps == 1
    
    def test_pruner_custom_params(self) -> None:
        """Test pruner with custom parameters."""
        pruner = MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=2,
        )
        
        assert pruner.n_startup_trials == 10
        assert pruner.n_warmup_steps == 5
        assert pruner.interval_steps == 2
    
    def test_no_prune_during_warmup(self) -> None:
        """Test that pruning doesn't happen during warmup."""
        pruner = MedianPruner(n_warmup_steps=10)
        
        all_values = {0: [(5, 0.5)]}
        
        result = pruner.should_prune(
            trial_id=1,
            step=5,
            value=0.9,
            all_values=all_values,
        )
        
        assert result is False
    
    def test_no_prune_without_enough_trials(self) -> None:
        """Test that pruning doesn't happen without enough comparison data."""
        pruner = MedianPruner(n_startup_trials=5)
        
        # Only 2 comparison trials
        all_values = {
            0: [(5, 0.5)],
            1: [(5, 0.6)],
        }
        
        result = pruner.should_prune(
            trial_id=2,
            step=5,
            value=0.9,
            all_values=all_values,
        )
        
        assert result is False


class TestPercentilePruner:
    """Tests for PercentilePruner."""
    
    def test_pruner_creation(self) -> None:
        """Test pruner creation."""
        pruner = PercentilePruner(percentile=25.0)
        
        assert pruner.percentile == 25.0
        assert pruner.n_startup_trials == 5


class TestTuningTrial:
    """Tests for TuningTrial dataclass."""
    
    def test_trial_creation(self) -> None:
        """Test trial creation."""
        trial = TuningTrial(
            trial_id=1,
            config={"lr": 0.001},
            score=0.5,
            status="COMPLETE",
            duration_seconds=10.5,
        )
        
        assert trial.trial_id == 1
        assert trial.config["lr"] == 0.001
        assert trial.score == 0.5
        assert trial.status == "COMPLETE"


class TestTuningResult:
    """Tests for TuningResult dataclass."""
    
    def test_result_creation(self) -> None:
        """Test result creation."""
        trials = [
            TuningTrial(trial_id=0, config={"lr": 0.001}, score=0.5, status="COMPLETE", duration_seconds=10.0),
            TuningTrial(trial_id=1, config={"lr": 0.01}, score=0.3, status="COMPLETE", duration_seconds=12.0),
        ]
        
        result = TuningResult(
            best_config={"lr": 0.01},
            best_score=0.3,
            best_trial_id=1,
            trials=trials,
            n_trials=2,
            n_completed=2,
            n_pruned=0,
            optimization_history=[0.5, 0.3],
        )
        
        assert result.best_score == 0.3
        assert result.n_trials == 2
    
    def test_to_dict(self) -> None:
        """Test exporting to dictionary."""
        trials = [
            TuningTrial(trial_id=0, config={"lr": 0.001}, score=0.5, status="COMPLETE", duration_seconds=10.0),
        ]
        
        result = TuningResult(
            best_config={"lr": 0.001},
            best_score=0.5,
            best_trial_id=0,
            trials=trials,
            n_trials=1,
            n_completed=1,
            n_pruned=0,
            optimization_history=[0.5],
        )
        
        d = result.to_dict()
        
        assert d["best_score"] == 0.5
        assert d["n_trials"] == 1
        assert len(d["trials"]) == 1
    
    def test_save_and_load(self) -> None:
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            
            trials = [
                TuningTrial(trial_id=0, config={"lr": 0.001}, score=0.5, status="COMPLETE", duration_seconds=10.0),
            ]
            
            result = TuningResult(
                best_config={"lr": 0.001},
                best_score=0.5,
                best_trial_id=0,
                trials=trials,
                n_trials=1,
                n_completed=1,
                n_pruned=0,
                optimization_history=[0.5],
            )
            
            result.save(path)
            
            loaded = TuningResult.load(path)
            
            assert loaded.best_score == 0.5
            assert loaded.best_config == {"lr": 0.001}


class TestCreateSearchSpace:
    """Tests for create_search_space factory function."""
    
    def test_create_from_config(self) -> None:
        """Test creating search space from config dict."""
        config = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "hidden_units": {"type": "int", "low": 32, "high": 256},
            "activation": {"type": "categorical", "choices": ["relu", "tanh"]},
            "use_dropout": {"type": "bool"},
        }
        
        space = create_search_space(config)
        
        assert len(space) == 4
        assert "learning_rate" in space
        assert "hidden_units" in space
        assert "activation" in space
        assert "use_dropout" in space
