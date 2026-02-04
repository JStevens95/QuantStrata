"""
Portfolio Optimisation Pipeline.

Orchestrated pipeline for portfolio construction and optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.orchestrator.core.context import PipelineContext
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step


@dataclass
class PortfolioOptimisationConfig:
    """Configuration for portfolio optimisation pipeline."""
    
    # Optimisation method
    method: str = "mean_variance"  # "mean_variance", "risk_parity", "black_litterman"
    
    # Mean-variance settings
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    risk_free_rate: float = 0.02
    
    # Constraints
    long_only: bool = True
    max_weight: Optional[float] = 0.3
    min_weight: Optional[float] = None
    
    # Risk parity settings
    risk_budgets: Optional[List[float]] = None
    leverage: float = 1.0
    
    # Black-Litterman settings
    risk_aversion: float = 2.5
    tau: float = 0.05
    views: List[Tuple[Any, float]] = field(default_factory=list)
    view_confidences: List[float] = field(default_factory=list)
    
    # Covariance estimation
    covariance_method: str = "shrinkage"  # "sample", "ewm", "shrinkage"
    shrinkage_target: str = "constant_correlation"
    ewm_halflife: int = 60
    
    # Output
    output_dir: str = "./portfolio_results"


class LoadMarketDataStep(Step):
    """Step to load market data for optimisation."""
    
    name = "load_market_data"
    
    def run(self, ctx: PipelineContext) -> None:
        # Load returns data from context or file
        returns = ctx.get("returns_data")
        market_caps = ctx.get("market_caps")
        expected_returns = ctx.get("expected_returns")
        
        if returns is None:
            raise ValueError("Returns data must be provided in context")
        
        ctx.set("returns", returns)
        ctx.set("n_assets", returns.shape[1])
        
        self.logger.info(f"Loaded returns data: {returns.shape}")


class EstimateCovarianceStep(Step):
    """Step to estimate covariance matrix."""
    
    name = "estimate_covariance"
    
    def run(self, ctx: PipelineContext) -> None:
        from src.portfolio.optimization import CovarianceEstimator, ShrinkageEstimator
        
        config = ctx.get("opt_config")
        returns = ctx.get("returns")
        
        if config.covariance_method == "sample":
            estimator = CovarianceEstimator()
            cov = estimator.sample(returns, annualize=True)
        
        elif config.covariance_method == "ewm":
            estimator = CovarianceEstimator()
            cov = estimator.ewm(returns, halflife=config.ewm_halflife)
        
        elif config.covariance_method == "shrinkage":
            estimator = ShrinkageEstimator(shrinkage_target=config.shrinkage_target)
            result = estimator.estimate(returns, annualize=True)
            cov = result.covariance
            ctx.set("shrinkage_intensity", result.shrinkage_intensity)
            self.logger.info(f"Shrinkage intensity: {result.shrinkage_intensity:.2%}")
        
        else:
            raise ValueError(f"Unknown covariance method: {config.covariance_method}")
        
        ctx.set("covariance", cov)
        self.logger.info(f"Estimated covariance matrix: {cov.shape}")


class ComputeExpectedReturnsStep(Step):
    """Step to compute expected returns."""
    
    name = "compute_expected_returns"
    
    def run(self, ctx: PipelineContext) -> None:
        from src.portfolio.optimization import BlackLittermanModel
        
        config = ctx.get("opt_config")
        returns = ctx.get("returns")
        cov = ctx.get("covariance")
        market_caps = ctx.get("market_caps")
        expected_returns = ctx.get("expected_returns")
        
        if config.method == "black_litterman":
            if market_caps is None:
                raise ValueError("Market caps required for Black-Litterman")
            
            bl = BlackLittermanModel(
                market_caps=market_caps,
                covariance=cov,
                risk_aversion=config.risk_aversion,
                tau=config.tau,
                risk_free_rate=config.risk_free_rate,
            )
            
            if config.views:
                result = bl.posterior(
                    views=config.views,
                    confidences=config.view_confidences,
                )
                expected_returns = result.posterior_returns
                ctx.set("bl_result", result)
            else:
                expected_returns = bl.equilibrium_returns
            
            ctx.set("expected_returns", expected_returns)
            self.logger.info("Computed Black-Litterman returns")
        
        elif expected_returns is None:
            # Use historical mean returns
            expected_returns = np.mean(returns, axis=0) * 252
            ctx.set("expected_returns", expected_returns)
            self.logger.info("Using historical mean returns")
        
        else:
            ctx.set("expected_returns", expected_returns)


class OptimisePortfolioStep(Step):
    """Step to run portfolio optimisation."""
    
    name = "optimise_portfolio"
    
    def run(self, ctx: PipelineContext) -> None:
        from src.portfolio.optimization import (
            MeanVarianceOptimizer,
            MVConstraints,
            RiskParityOptimizer,
        )
        
        config = ctx.get("opt_config")
        cov = ctx.get("covariance")
        expected_returns = ctx.get("expected_returns")
        
        if config.method in ["mean_variance", "black_litterman"]:
            constraints = MVConstraints(
                long_only=config.long_only,
                max_weight=config.max_weight,
                min_weight=config.min_weight,
            )
            
            optimizer = MeanVarianceOptimizer(risk_free_rate=config.risk_free_rate)
            
            result = optimizer.optimize(
                expected_returns=expected_returns,
                covariance=cov,
                target_return=config.target_return,
                target_volatility=config.target_volatility,
                constraints=constraints,
            )
            
            ctx.set("opt_result", result)
            self.logger.info(
                f"Mean-variance optimisation: "
                f"return={result.expected_return:.2%}, "
                f"vol={result.volatility:.2%}, "
                f"sharpe={result.sharpe_ratio:.2f}"
            )
        
        elif config.method == "risk_parity":
            optimizer = RiskParityOptimizer()
            
            risk_budgets = None
            if config.risk_budgets:
                risk_budgets = np.array(config.risk_budgets)
            
            result = optimizer.optimize(
                covariance=cov,
                risk_budgets=risk_budgets,
                leverage=config.leverage,
                long_only=config.long_only,
            )
            
            ctx.set("opt_result", result)
            self.logger.info(
                f"Risk parity optimisation: "
                f"vol={result.volatility:.2%}"
            )
        
        else:
            raise ValueError(f"Unknown method: {config.method}")


class ComputeEfficientFrontierStep(Step):
    """Step to compute efficient frontier (optional)."""
    
    name = "compute_efficient_frontier"
    
    def run(self, ctx: PipelineContext) -> None:
        from src.portfolio.optimization import MeanVarianceOptimizer, MVConstraints
        
        config = ctx.get("opt_config")
        
        if config.method != "mean_variance":
            self.logger.info("Skipping efficient frontier (not mean-variance)")
            return
        
        cov = ctx.get("covariance")
        expected_returns = ctx.get("expected_returns")
        
        constraints = MVConstraints(
            long_only=config.long_only,
            max_weight=config.max_weight,
        )
        
        optimizer = MeanVarianceOptimizer(risk_free_rate=config.risk_free_rate)
        
        frontier = optimizer.efficient_frontier(
            expected_returns=expected_returns,
            covariance=cov,
            n_points=20,
            constraints=constraints,
        )
        
        ctx.set("efficient_frontier", frontier)
        self.logger.info(f"Computed efficient frontier with {len(frontier)} points")


class SaveResultsStep(Step):
    """Step to save optimisation results."""
    
    name = "save_results"
    
    def run(self, ctx: PipelineContext) -> None:
        import json
        
        config = ctx.get("opt_config")
        result = ctx.get("opt_result")
        
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output = {
            "method": config.method,
            "weights": result.weights.tolist(),
        }
        
        if hasattr(result, "expected_return"):
            output["expected_return"] = result.expected_return
            output["volatility"] = result.volatility
            output["sharpe_ratio"] = result.sharpe_ratio
        
        if hasattr(result, "risk_contributions"):
            output["risk_contributions"] = result.risk_contributions.tolist()
        
        with open(output_dir / "portfolio.json", "w") as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"Saved results to {output_dir}")


def create_portfolio_optimisation_pipeline(
    config: PortfolioOptimisationConfig,
) -> Pipeline:
    """
    Create portfolio optimisation pipeline.
    
    Parameters
    ----------
    config : PortfolioOptimisationConfig
        Pipeline configuration.
        
    Returns
    -------
    Pipeline
        Configured pipeline.
    """
    pipeline = Pipeline(
        name="portfolio_optimisation",
        steps=[
            LoadMarketDataStep(),
            EstimateCovarianceStep(),
            ComputeExpectedReturnsStep(),
            OptimisePortfolioStep(),
            ComputeEfficientFrontierStep(),
            SaveResultsStep(),
        ],
    )
    
    pipeline.context.set("opt_config", config)
    
    return pipeline


__all__ = [
    "PortfolioOptimisationConfig",
    "create_portfolio_optimisation_pipeline",
]
