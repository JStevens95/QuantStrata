"""
Risk Measures for Deep Hedging

This module provides risk measures used as objective functions in deep hedging
training. The hedging policy is optimised to minimise ρ(-P&L), where ρ is a
risk measure.

Theory
------
A risk measure ρ: L^p(Ω) → ℝ quantifies the "risk" of a random variable (loss).
Key properties:

1. **Coherent Risk Measures** satisfy:
   - Monotonicity: X ≤ Y ⟹ ρ(X) ≤ ρ(Y)
   - Translation invariance: ρ(X + c) = ρ(X) + c
   - Positive homogeneity: ρ(λX) = λρ(X) for λ > 0
   - Subadditivity: ρ(X + Y) ≤ ρ(X) + ρ(Y) (diversification benefit)

2. **Convex Risk Measures** relax positive homogeneity to convexity.

Risk measures implemented:
- VarianceRisk: ρ(L) = Var(L) — penalises P&L uncertainty
- MeanVarianceRisk: ρ(L) = E[L] + λ·Var(L) — mean-variance trade-off
- CVaRRisk: Expected Shortfall — average of worst α% outcomes
- EntropicRisk: ρ(L) = (1/γ)·log E[e^{γL}] — exponential utility certainty equivalent

All risk measures are designed to be:
- Differentiable (for gradient-based training)
- Vectorised (efficient batch computation)
- Compatible with PyTorch/JAX autograd

References
----------
- Rockafellar & Uryasev (2000) "Optimization of Conditional Value-at-Risk"
- Föllmer & Schied (2004) "Stochastic Finance" (coherent risk measures)
- docs/reference/deep_hedging/theory.md Section 6
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

import numpy as np


class RiskMeasure(ABC):
    """
    Abstract base class for risk measures.
    
    A risk measure maps a distribution of losses (or P&L) to a scalar risk value.
    In deep hedging, we minimise ρ(-P&L) to find the optimal hedging policy.
    
    Convention
    ----------
    - Input: array of losses (positive = bad) or P&L (positive = good)
    - Output: scalar risk value (lower = better for minimisation)
    
    For hedging:
    - Compute P&L for each path
    - Compute ρ(-P&L) as the loss
    - Minimise over policy parameters
    
    Example
    -------
    >>> risk = MeanVarianceRisk(risk_aversion=0.5)
    >>> pnl = np.array([10, -5, 3, -2, 8])  # P&L for 5 paths
    >>> loss = risk.compute(-pnl)  # Risk of the negative P&L
    >>> print(f"Risk: {loss:.2f}")
    """
    
    @abstractmethod
    def compute(
        self,
        losses: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the risk measure for a sample of losses.
        
        Parameters
        ----------
        losses : ndarray, shape (n_samples,)
            Sample of losses. Positive values = losses, negative = gains.
        weights : ndarray, shape (n_samples,), optional
            Sample weights (for importance sampling). Default: equal weights.
        
        Returns
        -------
        float
            Scalar risk value.
        """
        pass
    
    def compute_with_gradient(
        self,
        losses: np.ndarray,
        dloss_dtheta: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        """
        Compute risk and its gradient w.r.t. parameters.
        
        This is used for training when the losses depend on policy parameters θ.
        By chain rule: dρ/dθ = (dρ/dL) · (dL/dθ)
        
        Parameters
        ----------
        losses : ndarray, shape (n_samples,)
            Sample of losses.
        dloss_dtheta : ndarray, shape (n_samples, n_params)
            Gradient of each loss w.r.t. policy parameters.
        weights : ndarray, shape (n_samples,), optional
            Sample weights.
        
        Returns
        -------
        risk : float
            Scalar risk value.
        gradient : ndarray, shape (n_params,)
            Gradient of risk w.r.t. parameters.
        
        Notes
        -----
        Default implementation uses finite differences. Subclasses can override
        for analytic gradients.
        """
        risk = self.compute(losses, weights)
        
        # Numerical gradient via finite differences
        n_samples, n_params = dloss_dtheta.shape
        eps = 1e-6
        gradient = np.zeros(n_params)
        
        for i in range(n_params):
            losses_plus = losses + eps * dloss_dtheta[:, i]
            losses_minus = losses - eps * dloss_dtheta[:, i]
            gradient[i] = (self.compute(losses_plus, weights) - 
                          self.compute(losses_minus, weights)) / (2 * eps)
        
        return risk, gradient
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the risk measure."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class VarianceRisk(RiskMeasure):
    """
    Variance risk measure: ρ(L) = Var(L).
    
    Mathematical Form
    -----------------
    ρ(L) = E[(L - E[L])²] = E[L²] - E[L]²
    
    Properties:
    - Penalises deviation from the mean (both upside and downside)
    - Not coherent (fails monotonicity)
    - Simple and widely used
    
    Use Case
    --------
    Minimising variance of hedging P&L gives a "stable" hedge:
    the P&L distribution is concentrated, even if mean is not optimal.
    
    Notes
    -----
    - Does not distinguish between gains and losses
    - Use MeanVarianceRisk for mean-variance optimisation
    """
    
    def compute(
        self,
        losses: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute variance of losses."""
        losses = np.asarray(losses).ravel()
        
        if weights is not None:
            weights = np.asarray(weights).ravel()
            weights = weights / weights.sum()  # Normalise
            mean = np.sum(weights * losses)
            variance = np.sum(weights * (losses - mean) ** 2)
        else:
            variance = float(np.var(losses))
        
        return float(variance)
    
    @property
    def name(self) -> str:
        return "Variance"


@dataclass
class MeanVarianceRisk(RiskMeasure):
    """
    Mean-variance risk measure: ρ(L) = E[L] + λ·Var(L).
    
    Mathematical Form
    -----------------
    ρ(L) = E[L] + λ·Var(L)
    
    where λ > 0 is the risk aversion parameter.
    
    Equivalently, this maximises:
    E[-L] - λ·Var(L) = E[P&L] - λ·Var(P&L)
    
    Parameters
    ----------
    risk_aversion : float
        Risk aversion parameter λ. Higher = more penalty on variance.
        - λ = 0: Risk-neutral (maximise expected P&L)
        - λ = 0.5: Moderate risk aversion
        - λ = 1.0: High risk aversion
    
    Properties
    ----------
    - Not coherent (violates monotonicity and positive homogeneity)
    - Quadratic utility interpretation: U(x) = x - λx²
    - Widely used in portfolio theory (Markowitz)
    
    Gradient
    --------
    dρ/dL = 1 + 2λ(L - E[L])
    
    Example
    -------
    >>> risk = MeanVarianceRisk(risk_aversion=0.5)
    >>> pnl = np.array([10, -5, 3, -2, 8])
    >>> risk.compute(-pnl)  # Minimise risk of negative P&L
    """
    
    risk_aversion: float = 0.5  # λ
    
    def __post_init__(self):
        if self.risk_aversion < 0:
            raise ValueError(f"risk_aversion must be non-negative, got {self.risk_aversion}")
    
    def compute(
        self,
        losses: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute mean + λ·variance of losses."""
        losses = np.asarray(losses).ravel()
        
        if weights is not None:
            weights = np.asarray(weights).ravel()
            weights = weights / weights.sum()
            mean = np.sum(weights * losses)
            variance = np.sum(weights * (losses - mean) ** 2)
        else:
            mean = float(np.mean(losses))
            variance = float(np.var(losses))
        
        return float(mean + self.risk_aversion * variance)
    
    def compute_with_gradient(
        self,
        losses: np.ndarray,
        dloss_dtheta: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        """
        Analytic gradient for mean-variance risk.
        
        dρ/dθ = E[dL/dθ] + λ·E[2(L - E[L])·dL/dθ]
              = E[(1 + 2λ(L - E[L]))·dL/dθ]
        """
        losses = np.asarray(losses).ravel()
        dloss_dtheta = np.asarray(dloss_dtheta)
        
        if weights is not None:
            weights = np.asarray(weights).ravel()
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(losses)) / len(losses)
        
        mean = np.sum(weights * losses)
        variance = np.sum(weights * (losses - mean) ** 2)
        risk = mean + self.risk_aversion * variance
        
        # Gradient weights: 1 + 2λ(L - E[L])
        grad_weights = 1.0 + 2.0 * self.risk_aversion * (losses - mean)
        
        # Weighted average of dL/dθ with these weights
        # gradient[j] = Σᵢ wᵢ·grad_weights[i]·dloss_dtheta[i,j]
        gradient = np.sum(
            weights[:, None] * grad_weights[:, None] * dloss_dtheta,
            axis=0
        )
        
        return float(risk), gradient
    
    @property
    def name(self) -> str:
        return f"MeanVariance(λ={self.risk_aversion})"
    
    def __repr__(self) -> str:
        return f"MeanVarianceRisk(risk_aversion={self.risk_aversion})"


@dataclass
class CVaRRisk(RiskMeasure):
    """
    Conditional Value-at-Risk (Expected Shortfall) risk measure.
    
    Mathematical Form
    -----------------
    CVaR_α(L) = E[L | L ≥ VaR_α(L)]
    
    where VaR_α(L) = inf{l : P(L ≤ l) ≥ α} is the α-quantile.
    
    Equivalently (Rockafellar-Uryasev):
    CVaR_α(L) = min_ν { ν + (1/(1-α))·E[(L - ν)⁺] }
    
    Parameters
    ----------
    alpha : float
        Confidence level α ∈ (0, 1). Typical values: 0.95, 0.99.
        CVaR_0.95 is the average of the worst 5% of outcomes.
    
    Properties
    ----------
    - Coherent risk measure (satisfies all four axioms)
    - Sensitive to tail risk
    - More conservative than VaR
    - Convex in losses (suitable for optimisation)
    
    Gradient
    --------
    dCVaR/dL = (1/(1-α))·1_{L ≥ VaR_α(L)}
    
    Example
    -------
    >>> risk = CVaRRisk(alpha=0.95)  # Focus on worst 5%
    >>> losses = np.random.normal(0, 1, 10000)
    >>> cvar = risk.compute(losses)
    >>> print(f"CVaR_95%: {cvar:.3f}")  # ≈ 2.06 for standard normal
    
    Notes
    -----
    - Also called Expected Shortfall (ES) or Average Value-at-Risk (AVaR)
    - Regulatory standard for market risk (Basel III)
    - For normal distribution: CVaR_α ≈ φ(Φ⁻¹(α))/(1-α) · σ
    """
    
    alpha: float = 0.95  # Confidence level
    
    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
    
    def compute(
        self,
        losses: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute CVaR as average of losses above VaR.
        
        For sample computation:
        1. Sort losses
        2. Find VaR as the α-quantile
        3. Average the losses above VaR
        """
        losses = np.asarray(losses).ravel()
        n = len(losses)
        
        if weights is not None:
            # Weighted quantile computation
            weights = np.asarray(weights).ravel()
            weights = weights / weights.sum()
            
            # Sort by losses
            sorted_idx = np.argsort(losses)
            sorted_losses = losses[sorted_idx]
            sorted_weights = weights[sorted_idx]
            
            # Cumulative weights
            cum_weights = np.cumsum(sorted_weights)
            
            # Find VaR (smallest loss where cumulative weight ≥ alpha)
            var_idx = np.searchsorted(cum_weights, self.alpha)
            var_idx = min(var_idx, n - 1)
            var = sorted_losses[var_idx]
            
            # Average losses above VaR
            tail_mask = losses >= var
            tail_weights = weights * tail_mask
            if tail_weights.sum() > 0:
                cvar = np.sum(tail_weights * losses) / tail_weights.sum()
            else:
                cvar = var
        else:
            # Unweighted: simple percentile
            var = np.percentile(losses, 100 * self.alpha)
            tail_losses = losses[losses >= var]
            if len(tail_losses) > 0:
                cvar = float(np.mean(tail_losses))
            else:
                cvar = var
        
        return float(cvar)
    
    def compute_var(
        self,
        losses: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Value-at-Risk (the α-quantile)."""
        losses = np.asarray(losses).ravel()
        
        if weights is not None:
            weights = np.asarray(weights).ravel()
            weights = weights / weights.sum()
            sorted_idx = np.argsort(losses)
            sorted_losses = losses[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cum_weights = np.cumsum(sorted_weights)
            var_idx = np.searchsorted(cum_weights, self.alpha)
            var_idx = min(var_idx, len(losses) - 1)
            return float(sorted_losses[var_idx])
        else:
            return float(np.percentile(losses, 100 * self.alpha))
    
    def compute_with_gradient(
        self,
        losses: np.ndarray,
        dloss_dtheta: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        """
        Compute CVaR and its gradient.
        
        Gradient (subgradient):
        dCVaR/dθ = (1/(1-α)) · E[dL/dθ · 1_{L ≥ VaR}]
        """
        losses = np.asarray(losses).ravel()
        dloss_dtheta = np.asarray(dloss_dtheta)
        n = len(losses)
        
        var = self.compute_var(losses, weights)
        cvar = self.compute(losses, weights)
        
        # Tail indicator
        tail_mask = (losses >= var).astype(float)
        
        if weights is not None:
            weights = np.asarray(weights).ravel()
            weights = weights / weights.sum()
            # Normalise by tail probability
            tail_prob = np.sum(weights * tail_mask)
        else:
            weights = np.ones(n) / n
            tail_prob = np.mean(tail_mask)
        
        if tail_prob > 0:
            # dCVaR/dθ = (1/tail_prob) · Σᵢ wᵢ·1_{Lᵢ≥VaR}·dLᵢ/dθ
            gradient = np.sum(
                weights[:, None] * tail_mask[:, None] * dloss_dtheta,
                axis=0
            ) / tail_prob
        else:
            gradient = np.zeros(dloss_dtheta.shape[1])
        
        return float(cvar), gradient
    
    @property
    def name(self) -> str:
        return f"CVaR({100*self.alpha:.0f}%)"
    
    def __repr__(self) -> str:
        return f"CVaRRisk(alpha={self.alpha})"


@dataclass
class EntropicRisk(RiskMeasure):
    """
    Entropic risk measure (exponential utility certainty equivalent).
    
    Mathematical Form
    -----------------
    ρ_γ(L) = (1/γ)·log E[exp(γL)]
    
    This is the certainty equivalent under exponential utility:
    U(x) = -exp(-γx)  (CARA utility)
    
    Parameters
    ----------
    risk_aversion : float
        Risk aversion parameter γ > 0. Higher = more risk-averse.
        - γ → 0: Risk-neutral (ρ → E[L])
        - γ = 1: Moderate risk aversion
        - γ → ∞: Worst-case (ρ → max L)
    
    Properties
    ----------
    - Coherent risk measure (satisfies all four axioms)
    - Very sensitive to tail risk (exponential penalty)
    - Additive for independent risks
    - Related to Kullback-Leibler divergence
    
    Gradient
    --------
    dρ/dL = exp(γL) / E[exp(γL)]
    
    Example
    -------
    >>> risk = EntropicRisk(risk_aversion=1.0)
    >>> losses = np.array([1, 2, 3, 10])  # One large loss
    >>> print(risk.compute(losses))  # Heavily penalises the 10
    
    Notes
    -----
    - Can overflow for large γ or large losses; use log-sum-exp trick
    - Multiplicative in sense: ρ(aL) = a·ρ(L) (positive homogeneity)
    - Used in robust control and information theory
    """
    
    risk_aversion: float = 1.0  # γ
    
    def __post_init__(self):
        if self.risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be positive, got {self.risk_aversion}")
    
    def compute(
        self,
        losses: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute entropic risk using log-sum-exp for numerical stability.
        
        ρ = (1/γ)·log E[exp(γL)]
          = (1/γ)·(max(γL) + log E[exp(γL - max(γL))])
        """
        losses = np.asarray(losses).ravel()
        gamma = self.risk_aversion
        
        # Log-sum-exp trick for stability
        scaled_losses = gamma * losses
        max_val = np.max(scaled_losses)
        
        if weights is not None:
            weights = np.asarray(weights).ravel()
            weights = weights / weights.sum()
            log_expectation = max_val + np.log(
                np.sum(weights * np.exp(scaled_losses - max_val))
            )
        else:
            log_expectation = max_val + np.log(
                np.mean(np.exp(scaled_losses - max_val))
            )
        
        entropic_risk = log_expectation / gamma
        return float(entropic_risk)
    
    def compute_with_gradient(
        self,
        losses: np.ndarray,
        dloss_dtheta: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        """
        Compute entropic risk and gradient.
        
        dρ/dθ = E[exp(γL)·dL/dθ] / E[exp(γL)]
        """
        losses = np.asarray(losses).ravel()
        dloss_dtheta = np.asarray(dloss_dtheta)
        gamma = self.risk_aversion
        
        # Compute risk (with stability)
        scaled_losses = gamma * losses
        max_val = np.max(scaled_losses)
        exp_terms = np.exp(scaled_losses - max_val)
        
        if weights is not None:
            weights = np.asarray(weights).ravel()
            weights = weights / weights.sum()
            Z = np.sum(weights * exp_terms)
            risk = (max_val + np.log(Z)) / gamma
            
            # Gradient weights: exp(γL)/Z (normalised)
            grad_weights = weights * exp_terms / Z
        else:
            Z = np.mean(exp_terms)
            risk = (max_val + np.log(Z)) / gamma
            
            grad_weights = exp_terms / (len(losses) * Z)
        
        # Gradient: Σᵢ grad_weights[i] · dL[i]/dθ
        gradient = np.sum(grad_weights[:, None] * dloss_dtheta, axis=0)
        
        return float(risk), gradient
    
    @property
    def name(self) -> str:
        return f"Entropic(γ={self.risk_aversion})"
    
    def __repr__(self) -> str:
        return f"EntropicRisk(risk_aversion={self.risk_aversion})"


def create_risk_measure(
    name: str,
    **kwargs,
) -> RiskMeasure:
    """
    Factory function to create risk measures by name.
    
    Parameters
    ----------
    name : str
        Risk measure name: "variance", "mean_variance", "cvar", "entropic".
    **kwargs : dict
        Parameters for the risk measure.
    
    Returns
    -------
    RiskMeasure
        Instantiated risk measure.
    
    Example
    -------
    >>> risk = create_risk_measure("cvar", alpha=0.95)
    >>> risk = create_risk_measure("mean_variance", risk_aversion=0.5)
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    
    if name_lower == "variance":
        return VarianceRisk()
    elif name_lower in ("mean_variance", "meanvariance"):
        return MeanVarianceRisk(**kwargs)
    elif name_lower in ("cvar", "es", "expected_shortfall"):
        return CVaRRisk(**kwargs)
    elif name_lower in ("entropic", "exponential"):
        return EntropicRisk(**kwargs)
    else:
        raise ValueError(
            f"Unknown risk measure: {name}. "
            f"Options: variance, mean_variance, cvar, entropic"
        )


__all__ = [
    "RiskMeasure",
    "VarianceRisk",
    "MeanVarianceRisk",
    "CVaRRisk",
    "EntropicRisk",
    "create_risk_measure",
]
