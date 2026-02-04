"""
Loss functions for Neural SDE training.

Implements:
- Score matching: Match the score (gradient of log-density)
- Moment matching: Match statistical moments
- Pathwise loss: Match path statistics

Example:
    from src.models.neural_sde.training.losses import MomentMatchingLoss
    
    loss_fn = MomentMatchingLoss(weights={"mean": 1.0, "var": 1.0, "skew": 0.5})
    loss = loss_fn.compute(predicted_paths, target_paths)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# =============================================================================
# Base Loss
# =============================================================================


class BaseLoss(ABC):
    """Abstract base for training losses."""
    
    @abstractmethod
    def compute(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute loss.
        
        Parameters
        ----------
        predicted : ndarray
            Model predictions.
        target : ndarray
            Target values.
        
        Returns
        -------
        float
            Loss value.
        """
        pass


# =============================================================================
# Score Matching Loss
# =============================================================================


class ScoreMatchingLoss(BaseLoss):
    """
    Score matching loss for density estimation.
    
    Instead of maximizing likelihood directly, matches the score
    (gradient of log-density), which doesn't require knowing
    the normalizing constant.
    
    Loss = E[||∇_x log p_θ(x) - ∇_x log p_data(x)||²]
    
    In practice, uses denoising score matching:
    Loss = E[||σ(x) - σ_θ(x̃)||²]
    
    where x̃ = x + noise.
    """
    
    def __init__(
        self,
        noise_scale: float = 0.01,
        n_noise_samples: int = 10,
    ) -> None:
        """
        Initialize score matching loss.
        
        Parameters
        ----------
        noise_scale : float
            Scale of noise perturbation.
        n_noise_samples : int
            Number of noise samples per data point.
        """
        self.noise_scale = noise_scale
        self.n_noise_samples = n_noise_samples
    
    def compute(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute score matching loss.
        
        Parameters
        ----------
        predicted : ndarray
            Predicted volatility values from model.
        target : ndarray
            Target volatility values from data.
        
        Returns
        -------
        float
            Loss value.
        """
        # Simple MSE between predicted and target volatilities
        # (In full implementation, this would be score matching)
        diff = predicted - target
        return float(np.mean(diff ** 2))
    
    def compute_denoising_loss(
        self,
        model_diffusion: callable,
        data_points: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """
        Compute denoising score matching loss.
        
        Parameters
        ----------
        model_diffusion : callable
            Model's diffusion function σ_θ(x, t).
        data_points : ndarray
            Data points x.
        times : ndarray
            Time points t.
        
        Returns
        -------
        float
            Denoising score matching loss.
        """
        n_points = len(data_points)
        total_loss = 0.0
        
        for _ in range(self.n_noise_samples):
            # Add noise
            noise = np.random.randn(n_points) * self.noise_scale
            perturbed = data_points + noise
            
            # Model prediction on perturbed data
            predicted_sigma = model_diffusion(perturbed, times)
            
            # Target score (direction toward clean data)
            target_score = -noise / (self.noise_scale ** 2)
            
            # Loss: difference in implied score
            implied_score = -noise / (predicted_sigma ** 2 + 1e-8)
            loss = np.mean((implied_score - target_score) ** 2)
            
            total_loss += loss
        
        return total_loss / self.n_noise_samples


# =============================================================================
# Moment Matching Loss
# =============================================================================


@dataclass
class MomentMatchingLoss(BaseLoss):
    """
    Moment matching loss.
    
    Matches statistical moments between model-generated
    paths and historical data:
    - Mean return
    - Variance
    - Skewness
    - Kurtosis
    - Autocorrelation
    
    Example:
        loss_fn = MomentMatchingLoss(
            weights={"mean": 1.0, "var": 2.0, "skew": 0.5}
        )
        loss = loss_fn.compute(model_paths, historical_paths)
    """
    
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "mean": 1.0,
            "var": 1.0,
            "skew": 0.5,
            "kurt": 0.5,
            "autocorr": 0.5,
        }
    )
    
    def compute(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute moment matching loss.
        
        Parameters
        ----------
        predicted : ndarray
            Model-generated paths of shape (n_paths, n_steps).
        target : ndarray
            Historical paths of shape (n_paths, n_steps).
        
        Returns
        -------
        float
            Weighted sum of moment errors.
        """
        # Compute returns
        pred_returns = np.diff(np.log(np.maximum(predicted, 1e-8)), axis=1)
        targ_returns = np.diff(np.log(np.maximum(target, 1e-8)), axis=1)
        
        loss = 0.0
        
        # Mean
        if "mean" in self.weights:
            pred_mean = np.mean(pred_returns)
            targ_mean = np.mean(targ_returns)
            loss += self.weights["mean"] * (pred_mean - targ_mean) ** 2
        
        # Variance
        if "var" in self.weights:
            pred_var = np.var(pred_returns)
            targ_var = np.var(targ_returns)
            # Relative error for variance
            rel_err = (pred_var - targ_var) / (targ_var + 1e-8)
            loss += self.weights["var"] * rel_err ** 2
        
        # Skewness
        if "skew" in self.weights:
            pred_skew = self._skewness(pred_returns.flatten())
            targ_skew = self._skewness(targ_returns.flatten())
            loss += self.weights["skew"] * (pred_skew - targ_skew) ** 2
        
        # Kurtosis
        if "kurt" in self.weights:
            pred_kurt = self._kurtosis(pred_returns.flatten())
            targ_kurt = self._kurtosis(targ_returns.flatten())
            loss += self.weights["kurt"] * (pred_kurt - targ_kurt) ** 2
        
        # Autocorrelation (lag 1)
        if "autocorr" in self.weights:
            pred_ac = self._autocorr(pred_returns.flatten())
            targ_ac = self._autocorr(targ_returns.flatten())
            loss += self.weights["autocorr"] * (pred_ac - targ_ac) ** 2
        
        return float(loss)
    
    def _skewness(self, x: np.ndarray) -> float:
        """Compute skewness."""
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-8:
            return 0.0
        return float(np.mean(((x - m) / s) ** 3))
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis."""
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-8:
            return 0.0
        return float(np.mean(((x - m) / s) ** 4) - 3)
    
    def _autocorr(self, x: np.ndarray, lag: int = 1) -> float:
        """Compute autocorrelation at given lag."""
        if len(x) <= lag:
            return 0.0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


# =============================================================================
# Pathwise Loss
# =============================================================================


class PathwiseLoss(BaseLoss):
    """
    Pathwise loss for trajectory matching.
    
    Matches distributional properties of paths:
    - Terminal distribution
    - Maximum/minimum values
    - Time spent above/below barriers
    
    Example:
        loss_fn = PathwiseLoss()
        loss = loss_fn.compute(model_paths, historical_paths)
    """
    
    def __init__(
        self,
        weight_terminal: float = 1.0,
        weight_running_max: float = 0.5,
        weight_running_min: float = 0.5,
        n_quantiles: int = 10,
    ) -> None:
        """
        Initialize pathwise loss.
        
        Parameters
        ----------
        weight_terminal : float
            Weight for terminal distribution matching.
        weight_running_max : float
            Weight for running maximum distribution.
        weight_running_min : float
            Weight for running minimum distribution.
        n_quantiles : int
            Number of quantiles for distribution matching.
        """
        self.weight_terminal = weight_terminal
        self.weight_running_max = weight_running_max
        self.weight_running_min = weight_running_min
        self.n_quantiles = n_quantiles
        
        self.quantile_levels = np.linspace(0.05, 0.95, n_quantiles)
    
    def compute(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute pathwise loss.
        
        Parameters
        ----------
        predicted : ndarray
            Model paths of shape (n_paths, n_steps).
        target : ndarray
            Target paths of shape (n_paths, n_steps).
        
        Returns
        -------
        float
            Pathwise loss value.
        """
        loss = 0.0
        
        # Terminal distribution
        if self.weight_terminal > 0:
            pred_terminal = predicted[:, -1]
            targ_terminal = target[:, -1]
            loss += self.weight_terminal * self._quantile_loss(
                pred_terminal, targ_terminal
            )
        
        # Running maximum
        if self.weight_running_max > 0:
            pred_max = np.max(predicted, axis=1)
            targ_max = np.max(target, axis=1)
            loss += self.weight_running_max * self._quantile_loss(
                pred_max, targ_max
            )
        
        # Running minimum
        if self.weight_running_min > 0:
            pred_min = np.min(predicted, axis=1)
            targ_min = np.min(target, axis=1)
            loss += self.weight_running_min * self._quantile_loss(
                pred_min, targ_min
            )
        
        return float(loss)
    
    def _quantile_loss(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute quantile matching loss.
        
        Matches quantiles of predicted and target distributions.
        """
        pred_quantiles = np.quantile(predicted, self.quantile_levels)
        targ_quantiles = np.quantile(target, self.quantile_levels)
        
        # Normalized by target scale
        scale = np.std(target) + 1e-8
        diff = (pred_quantiles - targ_quantiles) / scale
        
        return float(np.mean(diff ** 2))


__all__ = [
    "BaseLoss",
    "ScoreMatchingLoss",
    "MomentMatchingLoss",
    "PathwiseLoss",
]
