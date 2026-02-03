"""
Unit tests for src.machine_learning.pipelines.training module.

Tests run_training() and TrainingLoop.
"""

import pytest
import numpy as np

# Skip entire module if TensorFlow is not available
tf = pytest.importorskip("tensorflow")

from src.machine_learning.pipelines.training import (
    run_training,
    TrainingLoop,
)
from src.machine_learning.core.types import TrainingConfig, TrainingResult
from src.machine_learning.core.protocols import KerasTrainableAdapter


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_trainable_model():
    """Create a simple Keras model wrapped as Trainable."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
    return KerasTrainableAdapter(model)


@pytest.fixture
def simple_data():
    """Simple linear data for training."""
    np.random.seed(42)
    X = np.random.randn(100, 4).astype(np.float32)
    y = (X.sum(axis=1, keepdims=True) + np.random.randn(100, 1) * 0.1).astype(np.float32)
    return X, y


@pytest.fixture
def simple_config():
    """Simple training config."""
    return TrainingConfig(
        epochs=10,
        learning_rate=0.01,
        batch_size=16,
        verbose=0,
    )


# =============================================================================
# TrainingLoop Tests
# =============================================================================


class TestTrainingLoop:
    """Tests for TrainingLoop class."""

    def test_initialization(self, simple_trainable_model, simple_config):
        """TrainingLoop initializes correctly."""
        loop = TrainingLoop(simple_trainable_model, simple_config)
        
        assert loop.model is simple_trainable_model
        assert loop.config is simple_config

    def test_run_returns_training_result(self, simple_trainable_model, simple_data, simple_config):
        """run() returns TrainingResult."""
        loop = TrainingLoop(simple_trainable_model, simple_config)
        features, targets = simple_data
        
        result = loop.run(features, targets)
        
        assert isinstance(result, TrainingResult)
        assert "loss" in result.history
        assert result.final_epoch > 0

    def test_run_with_validation_split(self, simple_trainable_model, simple_data):
        """run() splits data when validation_split > 0."""
        config = TrainingConfig(
            epochs=5,
            batch_size=16,
            validation_split=0.2,
            verbose=0,
        )
        loop = TrainingLoop(simple_trainable_model, config)
        features, targets = simple_data
        
        result = loop.run(features, targets)
        
        assert "val_loss" in result.history
        assert len(result.history["val_loss"]) == 5

    def test_run_with_explicit_validation(self, simple_trainable_model, simple_data):
        """run() accepts explicit validation data."""
        config = TrainingConfig(epochs=5, batch_size=16, verbose=0)
        loop = TrainingLoop(simple_trainable_model, config)
        features, targets = simple_data
        
        # Split manually
        val_features = features[:20]
        val_targets = targets[:20]
        train_features = features[20:]
        train_targets = targets[20:]
        
        result = loop.run(
            train_features, train_targets,
            val_features=val_features, val_targets=val_targets
        )
        
        assert "val_loss" in result.history
        assert len(result.history["val_loss"]) == 5

    def test_loss_decreases(self, simple_trainable_model, simple_data, simple_config):
        """Training loss should decrease over epochs."""
        loop = TrainingLoop(simple_trainable_model, simple_config)
        features, targets = simple_data
        
        result = loop.run(features, targets)
        
        # Loss should generally decrease (not strictly, but trend)
        losses = result.history["loss"]
        assert losses[-1] < losses[0]

    def test_best_epoch_tracked(self, simple_trainable_model, simple_data):
        """Best epoch is tracked correctly."""
        config = TrainingConfig(
            epochs=20,
            batch_size=16,
            validation_split=0.2,
            verbose=0,
        )
        loop = TrainingLoop(simple_trainable_model, config)
        features, targets = simple_data
        
        result = loop.run(features, targets)
        
        assert result.best_epoch >= 1
        assert result.best_epoch <= result.final_epoch

    def test_training_time_recorded(self, simple_trainable_model, simple_data, simple_config):
        """Training time is recorded."""
        loop = TrainingLoop(simple_trainable_model, simple_config)
        features, targets = simple_data
        
        result = loop.run(features, targets)
        
        assert result.training_time_seconds > 0

    def test_checkpointing(self, simple_trainable_model, simple_data, tmp_path):
        """Checkpoints are saved when configured."""
        config = TrainingConfig(
            epochs=5,
            batch_size=16,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            save_best_only=True,
            verbose=0,
            validation_split=0.2,
        )
        loop = TrainingLoop(simple_trainable_model, config)
        features, targets = simple_data
        
        result = loop.run(features, targets)
        
        # Should have at least one checkpoint
        assert len(result.checkpoints) >= 1
        # Best checkpoint should exist
        best_ckpts = [c for c in result.checkpoints if c.is_best]
        assert len(best_ckpts) >= 1

    def test_early_stopping(self, simple_trainable_model, simple_data):
        """Early stopping triggers when configured."""
        features, targets = simple_data
        
        config = TrainingConfig(
            epochs=1000,  # High max epochs
            batch_size=16,
            early_stopping_patience=5,
            validation_split=0.2,
            verbose=0,
        )
        loop = TrainingLoop(simple_trainable_model, config)
        
        result = loop.run(features, targets)
        
        # Should stop early (not reach 1000 epochs)
        assert result.final_epoch < 1000

    def test_custom_train_step_fn(self, simple_trainable_model, simple_data, simple_config):
        """Custom train_step_fn is used."""
        call_count = [0]
        
        def custom_train_step(features, targets):
            call_count[0] += 1
            return simple_trainable_model.train_step(features, targets)
        
        loop = TrainingLoop(simple_trainable_model, simple_config, train_step_fn=custom_train_step)
        features, targets = simple_data
        
        result = loop.run(features, targets)
        
        # Custom function should have been called multiple times
        assert call_count[0] > 0


# =============================================================================
# run_training Tests
# =============================================================================


class TestRunTraining:
    """Tests for run_training function."""

    def test_basic_usage(self, simple_trainable_model, simple_data, simple_config):
        """Basic training works."""
        features, targets = simple_data
        
        result = run_training(
            simple_trainable_model,
            features,
            targets,
            simple_config,
        )
        
        assert isinstance(result, TrainingResult)
        assert result.final_epoch == simple_config.epochs

    def test_with_validation_data(self, simple_trainable_model, simple_data, simple_config):
        """Training with explicit validation data."""
        features, targets = simple_data
        
        result = run_training(
            simple_trainable_model,
            features[20:],
            targets[20:],
            simple_config,
            val_features=features[:20],
            val_targets=targets[:20],
        )
        
        assert "val_loss" in result.history

    def test_returns_best_losses(self, simple_trainable_model, simple_data):
        """Result includes best loss values."""
        config = TrainingConfig(
            epochs=20,
            batch_size=16,
            validation_split=0.2,
            verbose=0,
        )
        features, targets = simple_data
        
        result = run_training(
            simple_trainable_model,
            features,
            targets,
            config,
        )
        
        assert result.best_train_loss > 0
        assert result.best_val_loss is not None
        assert result.best_val_loss > 0

    def test_config_stored_in_result(self, simple_trainable_model, simple_data, simple_config):
        """Config is stored in result."""
        features, targets = simple_data
        
        result = run_training(
            simple_trainable_model,
            features,
            targets,
            simple_config,
        )
        
        assert result.config is not None
        assert result.config.epochs == simple_config.epochs


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    def test_with_pricing_model(self):
        """Training works with PricingModel."""
        from src.machine_learning.models.pricing.model import create_mlp_pricer
        from src.machine_learning.data.pricing.build import build_pricing_data
        
        # Build model
        model = create_mlp_pricer(n_features=6, hidden_units=[32, 16])
        model.compile(optimizer='adam', loss='mse')
        adapter = KerasTrainableAdapter(model)
        
        # Build data
        data = build_pricing_data(n_samples=100, batch_size=16, seed=42)
        
        # Extract numpy arrays from dataset
        features = []
        targets = []
        for x, y in data.train_ds.unbatch():
            features.append(x.numpy())
            targets.append(y.numpy())
        features = np.stack(features)
        targets = np.stack(targets)
        
        # Train
        config = TrainingConfig(epochs=5, batch_size=16, verbose=0)
        result = run_training(adapter, features, targets, config)
        
        assert result.final_epoch == 5
        assert result.history["loss"][-1] < result.history["loss"][0]

    def test_full_pipeline_with_checkpoints(self, tmp_path):
        """Full pipeline with checkpointing."""
        from src.machine_learning.models.pricing.model import create_mlp_pricer
        
        # Simple data
        np.random.seed(42)
        X = np.random.randn(100, 6).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        
        # Model
        model = create_mlp_pricer(n_features=6, hidden_units=[32])
        model.compile(optimizer='adam', loss='mse')
        adapter = KerasTrainableAdapter(model)
        
        # Config with checkpointing
        config = TrainingConfig(
            epochs=10,
            batch_size=16,
            checkpoint_dir=str(tmp_path / "ckpts"),
            checkpoint_frequency=2,  # Save every 2 epochs
            validation_split=0.2,
            verbose=0,
        )
        
        result = run_training(adapter, X, y, config)
        
        # Should have checkpoints
        assert len(result.checkpoints) > 0
        
        # Result should be serializable
        result_dict = result.to_dict()
        assert "history" in result_dict
        assert "checkpoints" in result_dict
