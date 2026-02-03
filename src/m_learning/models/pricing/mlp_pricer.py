"""
Multi-Layer Perceptron (MLP) Pricer.

A flexible neural network for option pricing that can approximate
any pricing function given sufficient training data.

Architecture:
    Input → [Dense + BatchNorm + Dropout] × N → Output

Features:
    - Configurable depth and width
    - Batch normalization for training stability
    - Dropout for regularization
    - Skip connections (optional)
    - Automatic Greeks via tf.GradientTape

Usage:
    # Create model
    model = MLPPricer(hidden_units=[128, 64, 32], dropout_rate=0.1)
    
    # Or use factory function
    model = create_mlp_pricer(
        n_features=6,
        hidden_units=[128, 64, 32],
        use_batch_norm=True,
    )
    
    # Train
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(train_ds, validation_data=val_ds, epochs=100)
    
    # Predict with Greeks
    result = model.price_with_greeks(features)
    print(f"Price: {result['price']}, Delta: {result['delta']}")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras import layers

from src.m_learning.core.base import PricingModel
from src.m_learning.core.config import ModelConfig


@tf.keras.saving.register_keras_serializable(package="QuantStrata.m_learning")
class MLPPricer(PricingModel):
    """
    Multi-Layer Perceptron for option pricing.
    
    A flexible feedforward neural network that learns to approximate
    option prices from input features (spot, strike, vol, rate, expiry, type).
    
    Attributes:
        hidden_units: List of hidden layer sizes
        activation: Activation function ('relu', 'elu', 'swish', etc.)
        dropout_rate: Dropout rate (0 to disable)
        use_batch_norm: Whether to use batch normalization
        use_skip_connections: Whether to use residual connections
        kernel_regularizer: L2 regularization weight
    
    Example:
        model = MLPPricer(
            hidden_units=[128, 64, 32],
            dropout_rate=0.1,
            use_batch_norm=True,
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        history = model.fit(
            train_features, train_prices,
            validation_data=(val_features, val_prices),
            epochs=100,
            batch_size=256,
        )
    """
    
    def __init__(
        self,
        hidden_units: List[int] = [64, 32],
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        use_skip_connections: bool = False,
        kernel_regularizer: float = 0.0,
        output_activation: Optional[str] = None,
        name: str = "mlp_pricer",
        **kwargs
    ):
        """
        Initialize MLP Pricer.
        
        Args:
            hidden_units: List of hidden layer sizes
            activation: Activation function name
            dropout_rate: Dropout rate (0 to disable)
            use_batch_norm: Whether to use batch normalization
            use_skip_connections: Whether to use skip connections (residual)
            kernel_regularizer: L2 regularization weight (0 to disable)
            output_activation: Optional activation for output (e.g., 'softplus' for positive prices)
            name: Model name
        """
        super().__init__(name=name, output_greeks=True, **kwargs)
        
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_skip_connections = use_skip_connections
        self.kernel_regularizer_weight = kernel_regularizer
        self.output_activation = output_activation
        
        # Build regularizer
        regularizer = None
        if kernel_regularizer > 0:
            regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        
        # Build layers
        self.dense_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        self.skip_projections = []
        
        for i, units in enumerate(hidden_units):
            # Dense layer
            self.dense_layers.append(
                layers.Dense(
                    units,
                    activation=None,  # Apply activation after BN
                    kernel_regularizer=regularizer,
                    name=f"dense_{i}",
                )
            )
            
            # Batch normalization
            if use_batch_norm:
                self.bn_layers.append(layers.BatchNormalization(name=f"bn_{i}"))
            else:
                self.bn_layers.append(None)
            
            # Dropout
            if dropout_rate > 0:
                self.dropout_layers.append(layers.Dropout(dropout_rate, name=f"dropout_{i}"))
            else:
                self.dropout_layers.append(None)
            
            # Skip connection projection (if dimensions change)
            if use_skip_connections and i > 0:
                prev_units = hidden_units[i - 1]
                if prev_units != units:
                    self.skip_projections.append(
                        layers.Dense(units, use_bias=False, name=f"skip_proj_{i}")
                    )
                else:
                    self.skip_projections.append(None)
        
        # Output layer
        self.output_layer = layers.Dense(
            1,
            activation=output_activation,
            name="output",
        )
        
        # Activation layer
        self.activation_fn = layers.Activation(activation)
        
        # Update metadata
        self.update_metadata(
            hidden_units=hidden_units,
            activation=activation,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_skip_connections=use_skip_connections,
            kernel_regularizer=kernel_regularizer,
        )
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Feature tensor of shape [batch, n_features]
            training: Whether in training mode
        
        Returns:
            Predictions of shape [batch, 1]
        """
        x = inputs
        
        for i, (dense, bn, dropout) in enumerate(
            zip(self.dense_layers, self.bn_layers, self.dropout_layers)
        ):
            # Save for skip connection
            residual = x
            
            # Dense layer
            x = dense(x)
            
            # Batch normalization
            if bn is not None:
                x = bn(x, training=training)
            
            # Activation
            x = self.activation_fn(x)
            
            # Dropout
            if dropout is not None:
                x = dropout(x, training=training)
            
            # Skip connection (for layers after the first)
            if self.use_skip_connections and i > 0:
                if i - 1 < len(self.skip_projections) and self.skip_projections[i - 1] is not None:
                    residual = self.skip_projections[i - 1](residual)
                if residual.shape[-1] == x.shape[-1]:
                    x = x + residual
        
        # Output
        return self.output_layer(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "use_skip_connections": self.use_skip_connections,
            "kernel_regularizer": self.kernel_regularizer_weight,
            "output_activation": self.output_activation,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MLPPricer":
        """Create model from config."""
        return cls(**config)


def create_mlp_pricer(
    n_features: int = 6,
    hidden_units: List[int] = [128, 64, 32],
    activation: str = "relu",
    dropout_rate: float = 0.1,
    use_batch_norm: bool = True,
    use_skip_connections: bool = False,
    kernel_regularizer: float = 0.0,
    output_activation: Optional[str] = None,
    name: str = "mlp_pricer",
) -> MLPPricer:
    """
    Factory function to create an MLP pricer with recommended defaults.
    
    Args:
        n_features: Number of input features
        hidden_units: Hidden layer sizes
        activation: Activation function
        dropout_rate: Dropout rate
        use_batch_norm: Use batch normalization
        use_skip_connections: Use skip connections
        kernel_regularizer: L2 regularization weight
        output_activation: Output activation function
        name: Model name
    
    Returns:
        Compiled MLPPricer model
    
    Example:
        # Create model with good defaults
        model = create_mlp_pricer(n_features=6)
        
        # Train
        model.fit(X_train, y_train, epochs=100, validation_split=0.1)
    """
    model = MLPPricer(
        hidden_units=hidden_units,
        activation=activation,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        use_skip_connections=use_skip_connections,
        kernel_regularizer=kernel_regularizer,
        output_activation=output_activation,
        name=name,
    )
    
    # Build the model by calling it with dummy input
    dummy_input = tf.zeros((1, n_features))
    _ = model(dummy_input)
    
    return model


@tf.keras.saving.register_keras_serializable(package="QuantStrata.m_learning")
class ResidualMLPPricer(PricingModel):
    """
    Residual MLP Pricer with deep skip connections.
    
    Suitable for deeper networks where vanishing gradients
    could be an issue.
    
    Architecture:
        Input → Dense → [ResBlock] × N → Output
        
    Where each ResBlock is:
        x → Dense → BN → ReLU → Dense → BN → (+x) → ReLU
    """
    
    def __init__(
        self,
        n_blocks: int = 3,
        block_units: int = 64,
        dropout_rate: float = 0.1,
        name: str = "residual_mlp_pricer",
        **kwargs
    ):
        """
        Initialize Residual MLP Pricer.
        
        Args:
            n_blocks: Number of residual blocks
            block_units: Units per block
            dropout_rate: Dropout rate
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
        self.n_blocks = n_blocks
        self.block_units = block_units
        self.dropout_rate = dropout_rate
        
        # Input projection
        self.input_proj = layers.Dense(block_units, name="input_proj")
        
        # Residual blocks
        self.res_blocks = []
        for i in range(n_blocks):
            self.res_blocks.append(
                ResidualBlock(
                    units=block_units,
                    dropout_rate=dropout_rate,
                    name=f"res_block_{i}",
                )
            )
        
        # Output
        self.output_layer = layers.Dense(1, name="output")
        
        self.update_metadata(
            n_blocks=n_blocks,
            block_units=block_units,
            dropout_rate=dropout_rate,
        )
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass."""
        x = self.input_proj(inputs)
        
        for block in self.res_blocks:
            x = block(x, training=training)
        
        return self.output_layer(x)


@tf.keras.saving.register_keras_serializable(package="QuantStrata.m_learning")
class ResidualBlock(layers.Layer):
    """Residual block with two dense layers and skip connection."""
    
    def __init__(
        self,
        units: int,
        dropout_rate: float = 0.0,
        name: str = "res_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.dense1 = layers.Dense(units, name=f"{name}_dense1")
        self.bn1 = layers.BatchNormalization(name=f"{name}_bn1")
        self.dense2 = layers.Dense(units, name=f"{name}_dense2")
        self.bn2 = layers.BatchNormalization(name=f"{name}_bn2")
        
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.activation = layers.ReLU()
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass with skip connection."""
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        x = x + inputs
        
        return self.activation(x)
