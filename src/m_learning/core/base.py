"""
Base model classes for TensorFlow-native ML models.

This module provides abstract base classes that all ML models in the library
should inherit from. These ensure consistent interfaces for training,
evaluation, serialization, and inference.

Architecture:
    BaseModel (tf.keras.Model)
        └── PricingModel      — for option/derivative pricing
        └── CalibrationModel  — for model parameter calibration
        └── PortfolioModel    — for portfolio-level predictions

Usage:
    class MyPricer(PricingModel):
        def __init__(self, hidden_units: List[int] = [64, 32]):
            super().__init__(name="my_pricer")
            self.dense_layers = [tf.keras.layers.Dense(u, activation='relu') for u in hidden_units]
            self.output_layer = tf.keras.layers.Dense(1)

        def call(self, inputs, training=False):
            x = inputs
            for layer in self.dense_layers:
                x = layer(x)
            return self.output_layer(x)

Note:
    Requires TensorFlow 2.x. Install with: pip install tensorflow
"""
from __future__ import annotations

import json
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TensorFlow is required for the ML module. "
        "Install with: pip install tensorflow"
    )


@tf.keras.saving.register_keras_serializable(package="QuantStrata.m_learning")
class BaseModel(tf.keras.Model):
    """
    Abstract base class for all ML models in the library.
    
    Provides:
        - Consistent interface for training and inference
        - Built-in model metadata tracking
        - Standardized save/load with SavedModel format
        - Configuration management
    
    All models should inherit from this class (or its subclasses) to ensure
    compatibility with the library's training, evaluation, and inference pipelines.
    """
    
    def __init__(self, name: str = "base_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self._model_metadata: Dict[str, Any] = {
            "model_name": name,
            "model_class": self.__class__.__name__,
            "created_at": datetime.utcnow().isoformat(),
            "framework": "tensorflow",
            "framework_version": tf.__version__,
        }
        self._is_built = False
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        return self._model_metadata.copy()
    
    def update_metadata(self, **kwargs) -> None:
        """Update model metadata with additional key-value pairs."""
        self._model_metadata.update(kwargs)
    
    @abstractmethod
    def call(self, inputs: Any, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the model.
        
        Args:
            inputs: Model inputs (tensor or dict of tensors)
            training: Whether in training mode (affects dropout, batch norm, etc.)
        
        Returns:
            Model predictions as a tensor
        """
        raise NotImplementedError("Subclasses must implement call()")
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            "metadata": self._model_metadata,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseModel":
        """Create model from configuration dict."""
        metadata = config.pop("metadata", {})
        model = cls(**config)
        model._model_metadata.update(metadata)
        return model
    
    def summary_dict(self) -> Dict[str, Any]:
        """Return a dictionary summary of the model architecture."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "trainable_params": int(sum(tf.reduce_prod(w.shape) for w in self.trainable_weights)),
            "non_trainable_params": int(sum(tf.reduce_prod(w.shape) for w in self.non_trainable_weights)),
            "layers": [layer.name for layer in self.layers],
            "metadata": self.metadata,
        }


@tf.keras.saving.register_keras_serializable(package="QuantStrata.m_learning")
class PricingModel(BaseModel):
    """
    Base class for option/derivative pricing models.
    
    Pricing models take financial features (spot, strike, vol, rate, time, etc.)
    and output predicted prices or price-related quantities.
    
    Expected input features:
        - spot: Underlying price
        - strike: Strike price
        - volatility: Implied or realized volatility
        - rate: Risk-free interest rate
        - time_to_expiry: Time to maturity
        - option_type: Call (+1) or Put (-1)
        - (optional) additional features
    
    Output:
        - price: Predicted option price
        - (optional) greeks: Delta, gamma, vega, etc.
    """
    
    def __init__(self, name: str = "pricing_model", output_greeks: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_greeks = output_greeks
        self.update_metadata(model_type="pricing", output_greeks=output_greeks)
    
    @property
    def feature_names(self) -> List[str]:
        """Return expected feature names in order."""
        return ["spot", "strike", "volatility", "rate", "time_to_expiry", "option_type"]
    
    def price(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Compute option price (alias for call).
        
        Args:
            inputs: Feature tensor of shape [batch, n_features]
            training: Whether in training mode
        
        Returns:
            Predicted prices of shape [batch, 1] or [batch,]
        """
        return self.call(inputs, training=training)
    
    def price_with_greeks(
        self, inputs: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Compute price and Greeks using automatic differentiation.
        
        Uses TensorFlow's GradientTape to compute first and second order
        sensitivities with respect to input features.
        
        Args:
            inputs: Feature tensor of shape [batch, n_features]
        
        Returns:
            Dictionary with 'price', 'delta', 'gamma', 'vega', 'theta', 'rho'
        """
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(inputs)
                price = self.call(inputs, training=False)
            
            # First-order Greeks
            grads = tape1.gradient(price, inputs)
            delta = grads[:, 0:1]  # dP/dS (spot)
            vega = grads[:, 2:3]   # dP/d(vol)
            theta = -grads[:, 4:5]  # -dP/d(time) (negative convention)
            rho = grads[:, 3:4]    # dP/d(rate)
        
        # Second-order Greeks
        gamma = tape2.gradient(delta, inputs)[:, 0:1]  # d²P/dS²
        
        del tape1, tape2
        
        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }


@tf.keras.saving.register_keras_serializable(package="QuantStrata.m_learning")
class CalibrationModel(BaseModel):
    """
    Base class for model calibration networks.
    
    Calibration models solve the inverse problem: given market observables
    (e.g., option prices or implied volatilities), predict model parameters
    (e.g., Heston parameters, SABR parameters, etc.).
    
    Expected input features:
        - Market quotes (prices, implied vols, etc.)
        - Contract specifications (strikes, expiries, etc.)
    
    Output:
        - Model parameters (varies by target model)
    """
    
    def __init__(
        self,
        name: str = "calibration_model",
        target_model: str = "unknown",
        n_parameters: int = 1,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.target_model = target_model
        self.n_parameters = n_parameters
        self.update_metadata(
            model_type="calibration",
            target_model=target_model,
            n_parameters=n_parameters,
        )
    
    @property
    def parameter_names(self) -> List[str]:
        """Return names of output parameters (to be overridden by subclasses)."""
        return [f"param_{i}" for i in range(self.n_parameters)]
    
    def calibrate(self, market_data: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Calibrate model parameters from market data (alias for call).
        
        Args:
            market_data: Market observables tensor
            training: Whether in training mode
        
        Returns:
            Predicted model parameters
        """
        return self.call(market_data, training=training)
    
    def calibrate_with_bounds(
        self,
        market_data: tf.Tensor,
        lower_bounds: Optional[tf.Tensor] = None,
        upper_bounds: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Calibrate with parameter constraints via sigmoid squashing.
        
        Args:
            market_data: Market observables tensor
            lower_bounds: Lower bounds for each parameter
            upper_bounds: Upper bounds for each parameter
        
        Returns:
            Constrained parameter predictions
        """
        raw_params = self.call(market_data, training=False)
        
        if lower_bounds is not None and upper_bounds is not None:
            # Apply sigmoid squashing to constrain parameters
            lower = tf.convert_to_tensor(lower_bounds, dtype=tf.float32)
            upper = tf.convert_to_tensor(upper_bounds, dtype=tf.float32)
            return lower + (upper - lower) * tf.sigmoid(raw_params)
        
        return raw_params


@tf.keras.saving.register_keras_serializable(package="QuantStrata.m_learning")
class PortfolioModel(BaseModel):
    """
    Base class for portfolio-level ML models.
    
    Portfolio models operate on collections of trades/positions and may
    incorporate relational structure (graphs) and temporal dependencies.
    
    This is the base for the GNN-RNN hybrid and similar architectures.
    
    Expected inputs (dict):
        - trade_features: Per-trade feature matrix [batch, n_trades, n_features]
        - adjacency_matrix: Trade relationship graph [batch, n_trades, n_trades]
        - pnl_history: Historical P&L [batch, n_trades, n_timesteps]
        - target_indices: Indices of trades to predict [batch, n_targets]
    
    Output:
        - predictions: Per-trade predictions [batch, n_targets, output_dim]
    """
    
    def __init__(self, name: str = "portfolio_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.update_metadata(model_type="portfolio")
    
    @property
    def required_inputs(self) -> List[str]:
        """Return required input keys."""
        return ["trade_features", "adjacency_matrix"]
    
    def predict_portfolio(
        self,
        trade_features: tf.Tensor,
        adjacency_matrix: tf.Tensor,
        pnl_history: Optional[tf.Tensor] = None,
        target_indices: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Predict for portfolio (convenience method).
        
        Args:
            trade_features: Per-trade features [batch, n_trades, n_features]
            adjacency_matrix: Trade graph [batch, n_trades, n_trades]
            pnl_history: Historical P&L (optional) [batch, n_trades, n_timesteps]
            target_indices: Which trades to predict (optional)
            training: Whether in training mode
        
        Returns:
            Predictions tensor
        """
        inputs = {
            "trade_features": trade_features,
            "adjacency_matrix": adjacency_matrix,
        }
        if pnl_history is not None:
            inputs["pnl_history"] = pnl_history
        if target_indices is not None:
            inputs["target_indices"] = target_indices
        
        return self.call(inputs, training=training)
