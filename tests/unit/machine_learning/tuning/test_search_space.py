"""
Unit tests for hyperparameter tuning module.

Tests SearchSpace, ParameterDefinition, pruners, and tuning functions.
"""

import json
import random
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path for direct import
sys.path.insert(0, str(Path(__file__).parents[4]))

# Import directly from module file to avoid triggering tensorflow
from src.machine_learning.tuning.search_space import (
    SearchSpace,
    OptunaSearchSpace,
    ParameterType,
    ParameterDefinition,
    MedianPruner,
    PercentilePruner,
    TuningResult,
    TuningTrial,
    create_search_space,
)


class TestParameterDefinition:
    """Tests for ParameterDefinition dataclass."""
    
    def test_float_parameter(self):
        """Test float parameter creation."""
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
    
    def test_int_parameter(self):
        """Test integer parameter creation."""
        param = ParameterDefinition(
            name="hidden_units",
            param_type=ParameterType.INT,
            low=32,
            high=256,
        )
        
        assert param.name == "hidden_units"
        assert param.param_type == ParameterType.INT
    
    def test_categorical_parameter(self):
        """Test categorical parameter creation."""
        param = ParameterDefinition(
            name="activation",
            param_type=ParameterType.CATEGORICAL,
            choices=["relu", "tanh", "gelu"],
        )
        
        assert param.name == "activation"
        assert param.choices == ["relu", "tanh", "gelu"]
    
    def test_validation_float_missing_bounds(self):
        """Test that float parameters require bounds."""
        with pytest.raises(ValueError, match="low and high required"):
            ParameterDefinition(
                name="lr",
                param_type=ParameterType.FLOAT,
                low=None,
                high=1e-2,
            )
    
    def test_validation_categorical_missing_choices(self):
        """Test that categorical parameters require choices."""
        with pytest.raises(ValueError, match="choices required"):
            ParameterDefinition(
                name="opt",
                param_type=ParameterType.CATEGORICAL,
                choices=None,
            )
    
    def test_sample_float(self):
        """Test sampling float parameter."""
        param = ParameterDefinition(
            name="lr",
            param_type=ParameterType.FLOAT,
            low=0.0,
            high=1.0,
        )
        
        rng = random.Random(42)
        value = param.sample_random(rng)
        
        assert 0.0 <= value <= 1.0
        assert isinstance(value, float)
    
    def test_sample_float_log(self):
        """Test sampling float parameter in log scale."""
        param = ParameterDefinition(
            name="lr",
            param_type=ParameterType.FLOAT,
            low=1e-4,
            high=1e-1,
            log=True,
        )
        
        rng = random.Random(42)
        values = [param.sample_random(rng) for _ in range(100)]
        
        # All values should be in range
        assert all(1e-4 <= v <= 1e-1 for v in values)
    
    def test_sample_int(self):
        """Test sampling integer parameter."""
        param = ParameterDefinition(
            name="units",
            param_type=ParameterType.INT,
            low=32,
            high=128,
        )
        
        rng = random.Random(42)
        value = param.sample_random(rng)
        
        assert 32 <= value <= 128
        assert isinstance(value, int)
    
    def test_sample_categorical(self):
        """Test sampling categorical parameter."""
        param = ParameterDefinition(
            name="opt",
            param_type=ParameterType.CATEGORICAL,
            choices=["adam", "sgd", "rmsprop"],
        )
        
        rng = random.Random(42)
        value = param.sample_random(rng)
        
        assert value in ["adam", "sgd", "rmsprop"]


class TestSearchSpace:
    """Tests for SearchSpace class."""
    
    def test_empty_search_space(self):
        """Test empty search space."""
        space = SearchSpace()
        
        assert len(space) == 0
        assert space.parameters == []
        assert space.names == []
    
    def test_add_float(self):
        """Test adding float parameter."""
        space = SearchSpace()
        result = space.add_float("lr", 1e-4, 1e-2, log=True)
        
        assert result is space  # Check chaining
        assert len(space) == 1
        assert "lr" in space
        
        param = space.get("lr")
        assert param.param_type == ParameterType.FLOAT
        assert param.log is True
    
    def test_add_int(self):
        """Test adding integer parameter."""
        space = SearchSpace()
        space.add_int("units", 32, 256, log=False)
        
        assert len(space) == 1
        param = space.get("units")
        assert param.param_type == ParameterType.INT
    
    def test_add_categorical(self):
        """Test adding categorical parameter."""
        space = SearchSpace()
        space.add_categorical("activation", ["relu", "tanh"])
        
        param = space.get("activation")
        assert param.param_type == ParameterType.CATEGORICAL
        assert param.choices == ["relu", "tanh"]
    
    def test_add_bool(self):
        """Test adding boolean parameter."""
        space = SearchSpace()
        space.add_bool("use_dropout")
        
        param = space.get("use_dropout")
        assert param.param_type == ParameterType.BOOL
        assert param.choices == [True, False]
    
    def test_fluent_api(self):
        """Test fluent API for building search space."""
        space = (
            SearchSpace()
            .add_float("lr", 1e-4, 1e-2, log=True)
            .add_int("units", 32, 256)
            .add_categorical("opt", ["adam", "sgd"])
            .add_bool("dropout")
        )
        
        assert len(space) == 4
        assert space.names == ["lr", "units", "opt", "dropout"]
    
    def test_sample(self):
        """Test sampling configuration."""
        space = (
            SearchSpace()
            .add_float("lr", 0.001, 0.1)
            .add_int("units", 32, 64)
            .add_categorical("opt", ["adam", "sgd"])
        )
        
        config = space.sample(seed=42)
        
        assert "lr" in config
        assert "units" in config
        assert "opt" in config
        assert 0.001 <= config["lr"] <= 0.1
        assert 32 <= config["units"] <= 64
        assert config["opt"] in ["adam", "sgd"]
    
    def test_sample_reproducibility(self):
        """Test that sampling with same seed is reproducible."""
        space = (
            SearchSpace()
            .add_float("lr", 0.001, 0.1)
            .add_int("units", 32, 64)
        )
        
        config1 = space.sample(seed=42)
        config2 = space.sample(seed=42)
        
        assert config1 == config2
    
    def test_to_dict(self):
        """Test exporting search space to dictionary."""
        space = (
            SearchSpace()
            .add_float("lr", 1e-4, 1e-2, log=True)
            .add_int("units", 32, 256)
            .add_categorical("opt", ["adam", "sgd"])
        )
        
        d = space.to_dict()
        
        assert "lr" in d
        assert d["lr"]["type"] == "float"
        assert d["lr"]["log"] is True
        assert d["units"]["type"] == "int"
        assert d["opt"]["type"] == "categorical"
    
    def test_from_dict(self):
        """Test creating search space from dictionary."""
        config = {
            "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "units": {"type": "int", "low": 32, "high": 256},
            "opt": {"type": "categorical", "choices": ["adam", "sgd"]},
        }
        
        space = SearchSpace.from_dict(config)
        
        assert len(space) == 3
        assert space.get("lr").param_type == ParameterType.FLOAT
        assert space.get("lr").log is True
    
    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = (
            SearchSpace()
            .add_float("lr", 1e-4, 1e-2, log=True)
            .add_int("units", 32, 256)
            .add_categorical("opt", ["adam", "sgd"])
        )
        
        restored = SearchSpace.from_dict(original.to_dict())
        
        assert original.to_dict() == restored.to_dict()


class TestCreateSearchSpace:
    """Tests for create_search_space factory function."""
    
    def test_create_from_config(self):
        """Test creating search space from config dict."""
        config = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "hidden_units": {"type": "int", "low": 32, "high": 256},
            "activation": {"type": "categorical", "choices": ["relu", "tanh"]},
        }
        
        space = create_search_space(config)
        
        assert len(space) == 3
        assert space.get("learning_rate").log is True


class TestMedianPruner:
    """Tests for MedianPruner."""
    
    def test_pruner_creation(self):
        """Test creating median pruner."""
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=2,
        )
        
        assert pruner.n_startup_trials == 5
        assert pruner.n_warmup_steps == 10
        assert pruner.interval_steps == 2
    
    def test_no_prune_during_warmup(self):
        """Test that pruner doesn't prune during warmup."""
        pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=10)
        
        all_values = {0: [(5, 0.1)]}  # Another trial at step 5
        
        # Step 5 is within warmup
        assert not pruner.should_prune(1, 5, 0.9, all_values)
    
    def test_no_prune_not_enough_trials(self):
        """Test that pruner doesn't prune without enough comparison data."""
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        
        all_values = {0: [(10, 0.1)]}  # Only 1 other trial
        
        # Not enough trials for comparison
        assert not pruner.should_prune(1, 10, 0.9, all_values)
    
    def test_prune_below_median(self):
        """Test that pruner prunes trials below median."""
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
        
        # 5 completed trials at step 10 with values [0.1, 0.2, 0.3, 0.4, 0.5]
        # Median is 0.3
        all_values = {
            0: [(10, 0.1)],
            1: [(10, 0.2)],
            2: [(10, 0.3)],
            3: [(10, 0.4)],
            4: [(10, 0.5)],
        }
        
        # Value 0.8 is above median 0.3, should prune (minimizing)
        assert pruner.should_prune(5, 10, 0.8, all_values)
        
        # Value 0.2 is below median, should not prune
        assert not pruner.should_prune(5, 10, 0.2, all_values)


class TestPercentilePruner:
    """Tests for PercentilePruner."""
    
    def test_pruner_creation(self):
        """Test creating percentile pruner."""
        pruner = PercentilePruner(
            percentile=25.0,
            n_startup_trials=5,
        )
        
        assert pruner.percentile == 25.0
        assert pruner.n_startup_trials == 5


class TestTuningResult:
    """Tests for TuningResult dataclass."""
    
    def test_result_creation(self):
        """Test creating tuning result."""
        trials = [
            TuningTrial(
                trial_id=0,
                config={"lr": 0.001},
                score=0.5,
                status="COMPLETE",
                duration_seconds=10.0,
            ),
            TuningTrial(
                trial_id=1,
                config={"lr": 0.01},
                score=0.3,
                status="COMPLETE",
                duration_seconds=12.0,
            ),
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
        assert result.best_trial_id == 1
        assert len(result.trials) == 2
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        trials = [
            TuningTrial(
                trial_id=0,
                config={"lr": 0.001},
                score=0.5,
                status="COMPLETE",
                duration_seconds=10.0,
            ),
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
    
    def test_save_and_load(self):
        """Test saving and loading result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trials = [
                TuningTrial(
                    trial_id=0,
                    config={"lr": 0.001},
                    score=0.5,
                    status="COMPLETE",
                    duration_seconds=10.0,
                ),
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
            
            path = Path(tmpdir) / "result.json"
            result.save(path)
            
            assert path.exists()
            
            loaded = TuningResult.load(path)
            
            assert loaded.best_score == 0.5
            assert loaded.best_config == {"lr": 0.001}
            assert len(loaded.trials) == 1


class TestOptunaIntegration:
    """Tests for Optuna integration (requires optuna)."""
    
    @pytest.fixture
    def check_optuna(self):
        """Skip tests if optuna not installed."""
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not installed")
    
    def test_optuna_search_space(self, check_optuna):
        """Test OptunaSearchSpace wrapper."""
        import optuna
        
        space = (
            SearchSpace()
            .add_float("lr", 1e-4, 1e-2, log=True)
            .add_int("units", 32, 256)
            .add_categorical("opt", ["adam", "sgd"])
        )
        
        optuna_space = OptunaSearchSpace(space)
        
        # Create a mock trial
        study = optuna.create_study()
        trial = study.ask()
        
        config = optuna_space.sample(trial)
        
        assert "lr" in config
        assert "units" in config
        assert "opt" in config
    
    def test_median_pruner_to_optuna(self, check_optuna):
        """Test converting MedianPruner to Optuna pruner."""
        import optuna
        
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        optuna_pruner = pruner.to_optuna()
        
        assert isinstance(optuna_pruner, optuna.pruners.MedianPruner)
