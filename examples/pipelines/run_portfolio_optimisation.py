#!/usr/bin/env python3
"""
Example: Portfolio Optimisation Pipeline

Demonstrates portfolio construction using mean-variance,
risk parity, and Black-Litterman methods.
"""

import numpy as np
from pathlib import Path

from src.orchestrator.pipelines.portfolio.optimise_portfolio import (
    PortfolioOptimisationConfig,
    create_portfolio_optimisation_pipeline,
)


def generate_sample_data(n_assets: int = 5, n_obs: int = 252):
    """Generate sample market data."""
    np.random.seed(42)
    
    # Asset parameters
    annual_returns = np.array([0.08, 0.10, 0.12, 0.06, 0.09])[:n_assets]
    annual_vols = np.array([0.15, 0.20, 0.25, 0.10, 0.18])[:n_assets]
    
    # Correlation matrix
    corr = np.array([
        [1.0, 0.6, 0.4, 0.2, 0.5],
        [0.6, 1.0, 0.5, 0.3, 0.4],
        [0.4, 0.5, 1.0, 0.2, 0.3],
        [0.2, 0.3, 0.2, 1.0, 0.2],
        [0.5, 0.4, 0.3, 0.2, 1.0],
    ])[:n_assets, :n_assets]
    
    # Generate covariance from correlation
    std = np.diag(annual_vols / np.sqrt(252))
    cov = std @ corr @ std
    
    # Generate returns
    daily_returns = annual_returns / 252
    returns = np.random.multivariate_normal(daily_returns, cov, n_obs)
    
    # Market caps (billions)
    market_caps = np.array([500, 400, 300, 200, 150])[:n_assets] * 1e9
    
    return returns, market_caps, annual_returns


def run_mean_variance_example():
    """Run mean-variance optimisation example."""
    print("\n" + "=" * 60)
    print("Mean-Variance Optimisation")
    print("=" * 60)
    
    returns, market_caps, expected_returns = generate_sample_data()
    
    config = PortfolioOptimisationConfig(
        method="mean_variance",
        risk_free_rate=0.02,
        long_only=True,
        max_weight=0.4,
        covariance_method="shrinkage",
        output_dir="./output/portfolio_mv",
    )
    
    pipeline = create_portfolio_optimisation_pipeline(config)
    pipeline.context.set("returns_data", returns)
    pipeline.context.set("expected_returns", expected_returns)
    
    pipeline.run()
    
    result = pipeline.context.get("opt_result")
    print(f"\nOptimal weights: {result.weights.round(3)}")
    print(f"Expected return: {result.expected_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")


def run_risk_parity_example():
    """Run risk parity optimisation example."""
    print("\n" + "=" * 60)
    print("Risk Parity Optimisation")
    print("=" * 60)
    
    returns, market_caps, _ = generate_sample_data()
    
    config = PortfolioOptimisationConfig(
        method="risk_parity",
        leverage=1.0,
        covariance_method="shrinkage",
        output_dir="./output/portfolio_rp",
    )
    
    pipeline = create_portfolio_optimisation_pipeline(config)
    pipeline.context.set("returns_data", returns)
    
    pipeline.run()
    
    result = pipeline.context.get("opt_result")
    print(f"\nOptimal weights: {result.weights.round(3)}")
    print(f"Risk contributions: {result.risk_contributions.round(3)}")
    print(f"Portfolio volatility: {result.volatility:.2%}")


def run_black_litterman_example():
    """Run Black-Litterman optimisation example."""
    print("\n" + "=" * 60)
    print("Black-Litterman Optimisation")
    print("=" * 60)
    
    returns, market_caps, _ = generate_sample_data()
    
    # Define views
    views = [
        (0, 0.10),        # Asset 0 returns 10%
        ([1, 3], 0.02),   # Asset 1 outperforms Asset 3 by 2%
    ]
    confidences = [0.6, 0.5]
    
    config = PortfolioOptimisationConfig(
        method="black_litterman",
        risk_aversion=2.5,
        tau=0.05,
        views=views,
        view_confidences=confidences,
        long_only=True,
        max_weight=0.4,
        covariance_method="shrinkage",
        output_dir="./output/portfolio_bl",
    )
    
    pipeline = create_portfolio_optimisation_pipeline(config)
    pipeline.context.set("returns_data", returns)
    pipeline.context.set("market_caps", market_caps)
    
    pipeline.run()
    
    result = pipeline.context.get("opt_result")
    bl_result = pipeline.context.get("bl_result")
    
    print(f"\nPrior (equilibrium) returns: {bl_result.equilibrium_returns.round(3)}")
    print(f"Posterior returns: {bl_result.posterior_returns.round(3)}")
    print(f"\nOptimal weights: {result.weights.round(3)}")
    print(f"Expected return: {result.expected_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")


def main():
    """Run all portfolio optimisation examples."""
    print("=" * 60)
    print("Portfolio Optimisation Pipeline Examples")
    print("=" * 60)
    
    # Create output directory
    Path("./output").mkdir(exist_ok=True)
    
    # Run examples
    run_mean_variance_example()
    run_risk_parity_example()
    run_black_litterman_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
