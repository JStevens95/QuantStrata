"""
Unit tests for src.machine_learning.core.protocols module.

Tests Trainable protocol and KerasTrainableAdapter.
"""

import pytest
import numpy as np

# Skip entire module if TensorFlow is not available
tf = pytest.importorskip("tensorflow")

from src.machine_learning.core.protocols import (
    Trainable,
    KerasTrainableAdapter,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_keras_model():
    """Create a simple Keras model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


@pytest.fixture
def sample_features():
    """Sample features for testing."""
    return np.random.randn(10, 6).astype(np.float32)


@pytest.fixture
def sample_targets():
    """Sample targets for testing."""
    return np.random.randn(10, 1).astype(np.float32)


# =============================================================================
# Trainable Protocol Tests
# =============================================================================


class TestTrainableProtocol:
    """Tests for Trainable protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Trainable protocol is runtime checkable."""
        assert hasattr(Trainable, '__protocol_attrs__') or hasattr(Trainable, '__subclasshook__')

    def test_custom_class_conforms_to_protocol(self):
        """Custom class implementing required methods conforms to protocol."""
        
        class MyTrainable:
            def forward(self, inputs):
                return inputs * 2
            
            def compute_loss(self, y_true, y_pred):
                return float(np.mean((y_true - y_pred) ** 2))
            
            def get_parameters(self):
                return {"weights": [1, 2, 3]}
            
            def set_parameters(self, params):
                pass
        
        obj = MyTrainable()
        assert isinstance(obj, Trainable)

    def test_class_missing_methods_not_trainable(self):
        """Class missing required methods doesn't conform."""
        
        class Incomplete:
            def forward(self, inputs):
                return inputs
        
        obj = Incomplete()
        assert not isinstance(obj, Trainable)


# =============================================================================
# KerasTrainableAdapter Tests
# =============================================================================


class TestKerasTrainableAdapter:
    """Tests for KerasTrainableAdapter class."""

    def test_adapter_wraps_keras_model(self, simple_keras_model):
        """Adapter wraps a Keras model."""
        adapter = KerasTrainableAdapter(simple_keras_model)
        assert adapter.model is simple_keras_model

    def test_adapter_conforms_to_trainable(self, simple_keras_model):
        """Adapter conforms to Trainable protocol."""
        adapter = KerasTrainableAdapter(simple_keras_model)
        assert isinstance(adapter, Trainable)

    def test_forward_returns_predictions(self, simple_keras_model, sample_features):
        """forward() returns model predictions."""
        adapter = KerasTrainableAdapter(simple_keras_model)
        output = adapter.forward(sample_features)
        
        assert output.shape == (10, 1)
        assert isinstance(output, (np.ndarray, tf.Tensor))

    def test_compute_loss_with_custom_loss(self, simple_keras_model, sample_targets):
        """compute_loss uses custom loss function if provided."""
        def custom_loss(y_true, y_pred):
            return tf.reduce_mean(tf.abs(y_true - y_pred))  # MAE
        
        adapter = KerasTrainableAdapter(simple_keras_model, loss_fn=custom_loss)
        
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.5], [2.5], [3.5]])
        
        loss = adapter.compute_loss(y_true, y_pred)
        assert isinstance(loss, float)
        assert abs(loss - 0.5) < 1e-6  # MAE should be 0.5

    def test_compute_loss_with_model_loss(self, simple_keras_model):
        """compute_loss uses model's compiled loss."""
        adapter = KerasTrainableAdapter(simple_keras_model)
        
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.5], [2.5], [3.5]])
        
        loss = adapter.compute_loss(y_true, y_pred)
        assert isinstance(loss, float)
        # MSE of 0.5^2 = 0.25
        assert abs(loss - 0.25) < 1e-6

    def test_compute_loss_default_mse(self):
        """compute_loss defaults to MSE if no loss specified."""
        # Uncompiled model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(2,))
        ])
        adapter = KerasTrainableAdapter(model)
        
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[0.0], [0.0]])
        
        loss = adapter.compute_loss(y_true, y_pred)
        # MSE = (1^2 + 2^2) / 2 = 2.5
        assert abs(loss - 2.5) < 1e-6

    def test_get_parameters(self, simple_keras_model, sample_features):
        """get_parameters returns weights dict."""
        # Build model
        _ = simple_keras_model(sample_features)
        
        adapter = KerasTrainableAdapter(simple_keras_model)
        params = adapter.get_parameters()
        
        assert "weights" in params
        assert "weight_names" in params
        assert len(params["weights"]) == len(params["weight_names"])
        # Should have at least 4 weight arrays (2 dense layers × (weights + biases))
        assert len(params["weights"]) >= 4

    def test_set_parameters(self, simple_keras_model, sample_features):
        """set_parameters restores weights."""
        # Build model
        _ = simple_keras_model(sample_features)
        
        adapter = KerasTrainableAdapter(simple_keras_model)
        
        # Get original parameters
        original_params = adapter.get_parameters()
        original_pred = adapter.forward(sample_features)
        
        # Modify weights
        for w in simple_keras_model.weights:
            w.assign(tf.zeros_like(w))
        
        modified_pred = adapter.forward(sample_features)
        
        # Predictions should be different after zeroing weights
        assert not np.allclose(original_pred.numpy(), modified_pred.numpy())
        
        # Restore parameters
        adapter.set_parameters(original_params)
        restored_pred = adapter.forward(sample_features)
        
        # Predictions should match original
        np.testing.assert_array_almost_equal(
            original_pred.numpy(),
            restored_pred.numpy(),
            decimal=5,
        )

    def test_train_step(self, simple_keras_model, sample_features, sample_targets):
        """train_step performs gradient update."""
        adapter = KerasTrainableAdapter(simple_keras_model)
        
        # Get initial loss
        initial_pred = adapter.forward(sample_features)
        initial_loss = adapter.compute_loss(sample_targets, initial_pred)
        
        # Run a few training steps
        for _ in range(10):
            loss = adapter.train_step(sample_features, sample_targets)
        
        # Loss should decrease
        final_pred = adapter.forward(sample_features)
        final_loss = adapter.compute_loss(sample_targets, final_pred)
        
        assert final_loss < initial_loss

    def test_train_step_returns_loss(self, simple_keras_model, sample_features, sample_targets):
        """train_step returns scalar loss."""
        adapter = KerasTrainableAdapter(simple_keras_model)
        loss = adapter.train_step(sample_features, sample_targets)
        
        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative for MSE


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainableIntegration:
    """Integration tests for Trainable implementations."""

    def test_adapter_with_pricing_model(self):
        """Adapter works with PricingModel."""
        from src.machine_learning.core.base import PricingModel
        
        class SimplePricer(PricingModel):
            def __init__(self):
                super().__init__(name="test_pricer")
                self.dense = tf.keras.layers.Dense(1)
            
            def call(self, inputs, training=False):
                return self.dense(inputs)
        
        model = SimplePricer()
        model.compile(optimizer='adam', loss='mse')
        
        adapter = KerasTrainableAdapter(model)
        
        features = np.random.randn(5, 6).astype(np.float32)
        targets = np.random.randn(5, 1).astype(np.float32)
        
        # All protocol methods should work
        output = adapter.forward(features)
        assert output.shape == (5, 1)
        
        loss = adapter.compute_loss(targets, output)
        assert isinstance(loss, float)
        
        params = adapter.get_parameters()
        assert "weights" in params
        
        train_loss = adapter.train_step(features, targets)
        assert isinstance(train_loss, float)

    def test_training_loop_with_adapter(self):
        """Full training loop using adapter."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
        
        adapter = KerasTrainableAdapter(model)
        
        # Generate simple linear data
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float32)
        y = (X.sum(axis=1, keepdims=True) + np.random.randn(100, 1) * 0.1).astype(np.float32)
        
        # Training loop
        losses = []
        for epoch in range(50):
            # Simple batch
            loss = adapter.train_step(X, y)
            losses.append(loss)
        
        # Loss should decrease significantly
        assert losses[-1] < losses[0] * 0.5
