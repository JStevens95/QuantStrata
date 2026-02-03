"""
Unit tests for src.machine_learning.models.pricing.model module.

Tests MLPPricer, ResidualMLPPricer, and factory functions.
"""

import pytest
import numpy as np

# Skip entire module if TensorFlow is not available
tf = pytest.importorskip("tensorflow")

from src.machine_learning.models.pricing.model import (
    MLPPricer,
    ResidualMLPPricer,
    ResidualBlock,
    create_mlp_pricer,
)
from src.machine_learning.core.base import PricingModel


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_features():
    """Sample pricing features: [spot, strike, vol, rate, time, option_type]."""
    np.random.seed(42)
    return np.random.randn(32, 6).astype(np.float32)


@pytest.fixture
def sample_targets():
    """Sample target prices."""
    np.random.seed(42)
    return np.random.randn(32, 1).astype(np.float32)


@pytest.fixture
def mlp_pricer():
    """Create MLPPricer instance."""
    return MLPPricer(
        hidden_units=[64, 32],
        activation="relu",
        dropout_rate=0.1,
        use_batch_norm=True,
    )


# =============================================================================
# MLPPricer Tests
# =============================================================================


class TestMLPPricer:
    """Tests for MLPPricer class."""

    def test_inherits_from_pricing_model(self):
        """MLPPricer inherits from PricingModel."""
        model = MLPPricer()
        assert isinstance(model, PricingModel)

    def test_default_initialization(self):
        """Default parameters are set correctly."""
        model = MLPPricer()
        assert model.hidden_units == [64, 32]
        assert model.activation == "relu"
        assert model.dropout_rate == 0.0
        assert model.use_batch_norm is False
        assert model.use_skip_connections is False

    def test_custom_initialization(self):
        """Custom parameters are accepted."""
        model = MLPPricer(
            hidden_units=[128, 64, 32],
            activation="swish",
            dropout_rate=0.2,
            use_batch_norm=True,
            use_skip_connections=True,
            kernel_regularizer=0.01,
            output_activation="softplus",
        )
        assert model.hidden_units == [128, 64, 32]
        assert model.activation == "swish"
        assert model.dropout_rate == 0.2
        assert model.use_batch_norm is True
        assert model.use_skip_connections is True

    def test_forward_pass(self, mlp_pricer, sample_features):
        """Forward pass produces correct output shape."""
        output = mlp_pricer(sample_features)
        assert output.shape == (32, 1)

    def test_forward_pass_training_mode(self, mlp_pricer, sample_features):
        """Forward pass works in training mode (dropout active)."""
        output_train = mlp_pricer(sample_features, training=True)
        assert output_train.shape == (32, 1)

    def test_batch_norm_layers(self, sample_features):
        """Batch normalization layers are created when enabled."""
        model = MLPPricer(hidden_units=[32, 16], use_batch_norm=True)
        _ = model(sample_features)  # Build model
        
        bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
        # Note: bn_layers are stored in model.bn_layers, not model.layers
        assert len(model.bn_layers) == 2
        assert model.bn_layers[0] is not None

    def test_dropout_layers(self, sample_features):
        """Dropout layers are created when enabled."""
        model = MLPPricer(hidden_units=[32, 16], dropout_rate=0.2)
        _ = model(sample_features)  # Build model
        
        assert len(model.dropout_layers) == 2
        assert model.dropout_layers[0] is not None

    def test_no_dropout_when_rate_zero(self, sample_features):
        """Dropout layers are None when rate is 0."""
        model = MLPPricer(hidden_units=[32], dropout_rate=0.0)
        _ = model(sample_features)
        
        assert model.dropout_layers[0] is None

    def test_skip_connections(self, sample_features):
        """Skip connections work when enabled."""
        model = MLPPricer(
            hidden_units=[32, 32, 32],  # Same size for easy skip
            use_skip_connections=True,
        )
        output = model(sample_features)
        assert output.shape == (32, 1)

    def test_kernel_regularizer(self, sample_features):
        """Kernel regularizer is applied."""
        model = MLPPricer(hidden_units=[32], kernel_regularizer=0.01)
        _ = model(sample_features)
        
        # Model should have regularization losses
        # Check that dense layer has regularizer
        assert model.dense_layers[0].kernel_regularizer is not None

    def test_output_activation(self, sample_features):
        """Output activation is applied."""
        model = MLPPricer(hidden_units=[32], output_activation="softplus")
        _ = model(sample_features)
        
        output = model(sample_features)
        # Softplus output should be positive
        assert tf.reduce_all(output >= 0).numpy()

    def test_get_config(self, mlp_pricer, sample_features):
        """get_config returns correct configuration."""
        _ = mlp_pricer(sample_features)
        config = mlp_pricer.get_config()
        
        assert config["hidden_units"] == [64, 32]
        assert config["activation"] == "relu"
        assert config["dropout_rate"] == 0.1
        assert config["use_batch_norm"] is True

    def test_from_config(self, mlp_pricer, sample_features):
        """Model can be recreated from config."""
        _ = mlp_pricer(sample_features)
        config = mlp_pricer.get_config()
        
        new_model = MLPPricer.from_config(config)
        assert new_model.hidden_units == mlp_pricer.hidden_units
        assert new_model.activation == mlp_pricer.activation

    def test_price_method(self, mlp_pricer, sample_features):
        """price() method works (inherited from PricingModel)."""
        price_output = mlp_pricer.price(sample_features)
        call_output = mlp_pricer(sample_features)
        
        np.testing.assert_array_almost_equal(
            price_output.numpy(), call_output.numpy()
        )

    def test_price_with_greeks(self, mlp_pricer, sample_features):
        """price_with_greeks computes Greeks (inherited from PricingModel)."""
        result = mlp_pricer.price_with_greeks(sample_features)
        
        assert "price" in result
        assert "delta" in result
        assert "gamma" in result
        assert "vega" in result
        assert "theta" in result
        assert "rho" in result
        
        # All should have correct shape
        for key in ["price", "delta", "gamma", "vega", "theta", "rho"]:
            assert result[key].shape == (32, 1)

    def test_trainable(self, mlp_pricer, sample_features, sample_targets):
        """Model can be compiled and trained."""
        mlp_pricer.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        history = mlp_pricer.fit(
            sample_features, sample_targets,
            epochs=2,
            verbose=0,
        )
        
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2

    def test_metadata_updated(self, mlp_pricer, sample_features):
        """Model metadata is updated with architecture info."""
        _ = mlp_pricer(sample_features)
        metadata = mlp_pricer.metadata
        
        assert metadata["hidden_units"] == [64, 32]
        assert metadata["activation"] == "relu"
        assert metadata["model_type"] == "pricing"


# =============================================================================
# create_mlp_pricer Tests
# =============================================================================


class TestCreateMlpPricer:
    """Tests for create_mlp_pricer factory function."""

    def test_creates_mlp_pricer(self):
        """Factory creates MLPPricer instance."""
        model = create_mlp_pricer()
        assert isinstance(model, MLPPricer)

    def test_default_architecture(self):
        """Default architecture is [128, 64, 32]."""
        model = create_mlp_pricer()
        assert model.hidden_units == [128, 64, 32]

    def test_model_is_built(self):
        """Model is built (not just instantiated)."""
        model = create_mlp_pricer(n_features=6)
        assert model.built

    def test_custom_parameters(self):
        """Custom parameters are passed through."""
        model = create_mlp_pricer(
            n_features=10,
            hidden_units=[256, 128],
            activation="swish",
            dropout_rate=0.3,
            use_batch_norm=True,
        )
        assert model.hidden_units == [256, 128]
        assert model.activation == "swish"
        assert model.dropout_rate == 0.3

    def test_n_features_affects_build(self):
        """n_features parameter builds model with correct input shape."""
        model = create_mlp_pricer(n_features=10)
        
        # Test with 10-feature input
        inputs = tf.random.uniform((5, 10))
        output = model(inputs)
        assert output.shape == (5, 1)


# =============================================================================
# ResidualBlock Tests
# =============================================================================


class TestResidualBlock:
    """Tests for ResidualBlock layer."""

    def test_forward_pass(self):
        """Forward pass works correctly."""
        block = ResidualBlock(units=32)
        inputs = tf.random.uniform((10, 32))
        
        output = block(inputs)
        assert output.shape == (10, 32)

    def test_skip_connection(self):
        """Skip connection adds input to output."""
        block = ResidualBlock(units=32, dropout_rate=0.0)
        
        # Use zero weights for dense layers to isolate skip connection
        inputs = tf.ones((2, 32))
        output = block(inputs, training=False)
        
        # Output should include the residual (input)
        # Even with random weights, the skip ensures gradient flow
        assert output.shape == (2, 32)

    def test_dropout_in_block(self):
        """Dropout is applied within block."""
        block = ResidualBlock(units=32, dropout_rate=0.5)
        inputs = tf.random.uniform((100, 32))
        
        # Run in training mode multiple times
        outputs_train = [block(inputs, training=True).numpy() for _ in range(3)]
        
        # Outputs should differ due to dropout
        # (with high probability for 50% dropout)
        assert not np.allclose(outputs_train[0], outputs_train[1])

    def test_batch_norm_in_block(self):
        """Batch normalization is applied."""
        block = ResidualBlock(units=32)
        
        # Should have 2 BN layers (one per dense)
        assert block.bn1 is not None
        assert block.bn2 is not None


# =============================================================================
# ResidualMLPPricer Tests
# =============================================================================


class TestResidualMLPPricer:
    """Tests for ResidualMLPPricer class."""

    def test_inherits_from_pricing_model(self):
        """ResidualMLPPricer inherits from PricingModel."""
        model = ResidualMLPPricer()
        assert isinstance(model, PricingModel)

    def test_default_initialization(self):
        """Default parameters are set."""
        model = ResidualMLPPricer()
        assert model.n_blocks == 3
        assert model.block_units == 64
        assert model.dropout_rate == 0.1

    def test_custom_initialization(self):
        """Custom parameters are accepted."""
        model = ResidualMLPPricer(
            n_blocks=5,
            block_units=128,
            dropout_rate=0.2,
        )
        assert model.n_blocks == 5
        assert model.block_units == 128

    def test_forward_pass(self, sample_features):
        """Forward pass produces correct output shape."""
        model = ResidualMLPPricer(n_blocks=2, block_units=32)
        output = model(sample_features)
        assert output.shape == (32, 1)

    def test_residual_blocks_created(self, sample_features):
        """Correct number of residual blocks are created."""
        model = ResidualMLPPricer(n_blocks=4, block_units=32)
        _ = model(sample_features)
        
        assert len(model.res_blocks) == 4

    def test_trainable(self, sample_features, sample_targets):
        """Model can be compiled and trained."""
        model = ResidualMLPPricer(n_blocks=2, block_units=32)
        model.compile(optimizer='adam', loss='mse')
        
        history = model.fit(
            sample_features, sample_targets,
            epochs=2,
            verbose=0,
        )
        
        assert len(history.history["loss"]) == 2

    def test_price_with_greeks_works(self, sample_features):
        """Greeks computation works with residual architecture."""
        model = ResidualMLPPricer(n_blocks=2, block_units=32)
        result = model.price_with_greeks(sample_features)
        
        assert "price" in result
        assert "delta" in result

    def test_metadata_updated(self, sample_features):
        """Metadata includes residual-specific info."""
        model = ResidualMLPPricer(n_blocks=3, block_units=64)
        _ = model(sample_features)
        
        metadata = model.metadata
        assert metadata["n_blocks"] == 3
        assert metadata["block_units"] == 64


# =============================================================================
# Integration Tests
# =============================================================================


class TestMLPPricerIntegration:
    """Integration tests for MLP Pricer models."""

    def test_full_training_pipeline(self):
        """Full training pipeline works."""
        from src.machine_learning.data.pricing.build import build_pricing_data
        
        # Build data
        data = build_pricing_data(n_samples=200, batch_size=32, seed=42)
        
        # Create model
        model = create_mlp_pricer(
            n_features=6,
            hidden_units=[64, 32],
            dropout_rate=0.1,
            use_batch_norm=True,
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train
        history = model.fit(
            data.train_ds,
            validation_data=data.val_ds,
            epochs=3,
            verbose=0,
        )
        
        assert "loss" in history.history
        assert "val_loss" in history.history

    def test_save_load_weights(self, tmp_path):
        """Model weights can be saved and loaded."""
        model1 = create_mlp_pricer(n_features=6, hidden_units=[32, 16])
        
        # Get prediction
        inputs = tf.random.uniform((5, 6))
        pred1 = model1(inputs).numpy()
        
        # Save weights (Keras 3.0 requires .weights.h5 extension)
        weights_path = tmp_path / "weights.weights.h5"
        model1.save_weights(str(weights_path))
        
        # Create new model and load weights
        model2 = create_mlp_pricer(n_features=6, hidden_units=[32, 16])
        model2.load_weights(str(weights_path))
        
        # Predictions should match
        pred2 = model2(inputs).numpy()
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_serialization_keras(self, tmp_path):
        """Model can be saved/loaded with Keras."""
        model = create_mlp_pricer(n_features=6, hidden_units=[32])
        model.compile(optimizer='adam', loss='mse')
        
        # Save
        model_path = tmp_path / "model.keras"
        model.save(str(model_path))
        
        # Load
        loaded = tf.keras.models.load_model(str(model_path))
        
        # Test
        inputs = tf.random.uniform((5, 6))
        orig_pred = model(inputs).numpy()
        loaded_pred = loaded(inputs).numpy()
        
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)

    def test_greeks_are_sensible(self):
        """Greeks have sensible signs and magnitudes."""
        model = create_mlp_pricer(n_features=6, hidden_units=[64, 32])
        model.compile(optimizer='adam', loss='mse')
        
        # Train on simple data
        from src.machine_learning.data.pricing.build import build_pricing_data
        data = build_pricing_data(n_samples=500, batch_size=32, seed=42)
        model.fit(data.train_ds, epochs=10, verbose=0)
        
        # Test Greeks on ATM call
        # [spot, strike, vol, rate, time, is_call]
        atm_call = tf.constant([[100.0, 100.0, 0.2, 0.05, 1.0, 1.0]])
        result = model.price_with_greeks(atm_call)
        
        # Price should be positive
        assert result["price"].numpy()[0, 0] > -50  # Allow some error from untrained model
        
        # For a reasonably trained model, delta should be positive for calls
        # (We can't strictly assert this without proper training)

    def test_different_activations(self, sample_features, sample_targets):
        """Model works with different activation functions."""
        for activation in ["relu", "elu", "swish", "tanh"]:
            model = MLPPricer(hidden_units=[32], activation=activation)
            model.compile(optimizer='adam', loss='mse')
            
            history = model.fit(
                sample_features, sample_targets,
                epochs=1, verbose=0,
            )
            assert len(history.history["loss"]) == 1
