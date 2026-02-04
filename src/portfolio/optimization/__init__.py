"""
Portfolio Optimization Module.

Provides institutional-grade portfolio construction tools:
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- Black-Litterman Model
- Covariance Estimation
- Constraint Handling

Research Foundation:
- Markowitz (1952) "Portfolio Selection"
- Black & Litterman (1992)
- Maillard, Roncalli, Teiletche (2010) "On the Properties of Equally-Weighted Risk Contribution"

Example:
    from src.portfolio.optimization import (
        MeanVarianceOptimizer,
        RiskParityOptimizer,
        BlackLittermanModel,
    )
    
    # Mean-variance
    mv_optimizer = MeanVarianceOptimizer()
    weights = mv_optimizer.optimize(
        expected_returns=mu,
        covariance=cov,
        target_return=0.10,
    )
    
    # Risk parity
    rp_optimizer = RiskParityOptimizer()
    weights = rp_optimizer.optimize(covariance=cov)
    
    # Black-Litterman
    bl_model = BlackLittermanModel(market_caps=caps, covariance=cov)
    posterior_mu = bl_model.posterior_mean(views=views, view_confidences=conf)
"""

from src.portfolio.optimization.mean_variance import (
    MeanVarianceOptimizer,
    MVOptimizationResult,
    MVConstraints,
)
from src.portfolio.optimization.risk_parity import (
    RiskParityOptimizer,
    RiskParityResult,
)
from src.portfolio.optimization.black_litterman import (
    BlackLittermanModel,
    BlackLittermanResult,
)
from src.portfolio.optimization.covariance import (
    CovarianceEstimator,
    ShrinkageEstimator,
)

__all__ = [
    # Mean-Variance
    "MeanVarianceOptimizer",
    "MVOptimizationResult",
    "MVConstraints",
    # Risk Parity
    "RiskParityOptimizer",
    "RiskParityResult",
    # Black-Litterman
    "BlackLittermanModel",
    "BlackLittermanResult",
    # Covariance
    "CovarianceEstimator",
    "ShrinkageEstimator",
]
