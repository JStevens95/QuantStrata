"""
Pipeline: risk.compute_var

Compute Value-at-Risk using multiple methods.

Purpose
-------
Calculate VaR using historical, parametric, and Monte Carlo methods:
1. Load portfolio and market from state
2. Load historical returns data
3. Compute historical simulation VaR
4. Compute parametric (delta-normal) VaR
5. Compute Monte Carlo VaR
6. Compute Expected Shortfall (CVaR)
7. Compare methods and write report

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.portfolio.core import Portfolio


# =============================================================================
# Configuration Helpers
# =============================================================================

def _var_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'var' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("var", {})


# =============================================================================
# Pipeline Steps
# =============================================================================

@dataclass(frozen=True, slots=True)
class LoadPortfolioStep(Step):
    """Step 1: Load portfolio from state."""
    
    def run(self, ctx: Context) -> Context:
        if Keys.PORTFOLIO not in ctx.state:
            raise KeyError("Missing portfolio in state")
        if Keys.MARKET not in ctx.state:
            raise KeyError("Missing market snapshot in state")
        return ctx


@dataclass(frozen=True, slots=True)
class LoadHistoricalDataStep(Step):
    """Step 3: Load historical returns for VaR computation."""
    
    def run(self, ctx: Context) -> Context:
        var_cfg = _var_cfg(ctx.cfg)
        methods = var_cfg.get("methods", {})
        
        lookback = methods.get("historical", {}).get("lookback_days", 252)
        
        # Generate synthetic historical returns for demonstration
        # In production, load from market data provider
        np.random.seed(42)
        historical_returns = np.random.normal(0.0001, 0.015, lookback)
        
        ctx.put(Keys.HISTORICAL_RETURNS, historical_returns)
        
        if ctx.logger:
            ctx.logger.info("Loaded %d days of historical returns", len(historical_returns))
        
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeHistoricalVaRStep(Step):
    """Step 4: Historical simulation VaR."""
    
    def run(self, ctx: Context) -> Context:
        var_cfg = _var_cfg(ctx.cfg)
        confidence_levels = var_cfg.get("confidence_levels", [0.95, 0.99])
        
        returns = ctx.get(Keys.HISTORICAL_RETURNS)
        
        # Portfolio value (simplified)
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        pv = sum(1_000_000 * pos.quantity for pos in portfolio)  # Placeholder
        
        # Historical VaR at each confidence level
        historical_var = {}
        for alpha in confidence_levels:
            percentile = (1 - alpha) * 100
            var_return = np.percentile(returns, percentile)
            historical_var[alpha] = -var_return * abs(pv)
        
        ctx.put(Keys.HISTORICAL_VAR, historical_var)
        
        if ctx.logger:
            for alpha, var in historical_var.items():
                ctx.logger.info("Historical VaR (%.0f%%): %.2f", alpha * 100, var)
        
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeParametricVaRStep(Step):
    """Step 5: Parametric (delta-normal) VaR."""
    
    def run(self, ctx: Context) -> Context:
        from scipy.stats import norm
        
        var_cfg = _var_cfg(ctx.cfg)
        confidence_levels = var_cfg.get("confidence_levels", [0.95, 0.99])
        
        returns = ctx.get(Keys.HISTORICAL_RETURNS)
        
        # Estimate parameters
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Portfolio value
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        pv = sum(1_000_000 * pos.quantity for pos in portfolio)
        
        # Parametric VaR
        parametric_var = {}
        for alpha in confidence_levels:
            z = norm.ppf(1 - alpha)
            var_value = -(mu + z * sigma) * abs(pv)
            parametric_var[alpha] = var_value
        
        ctx.put(Keys.PARAMETRIC_VAR, parametric_var)
        
        if ctx.logger:
            for alpha, var in parametric_var.items():
                ctx.logger.info("Parametric VaR (%.0f%%): %.2f", alpha * 100, var)
        
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeMonteCarloVaRStep(Step):
    """Step 6: Monte Carlo VaR."""
    
    def run(self, ctx: Context) -> Context:
        var_cfg = _var_cfg(ctx.cfg)
        methods = var_cfg.get("methods", {})
        mc_cfg = methods.get("monte_carlo", {})
        
        if not mc_cfg.get("enabled", True):
            if ctx.logger:
                ctx.logger.info("Monte Carlo VaR skipped (not enabled)")
            return ctx
        
        n_sims = mc_cfg.get("n_simulations", 10000)
        confidence_levels = var_cfg.get("confidence_levels", [0.95, 0.99])
        
        # Parameters from historical data
        returns = ctx.get(Keys.HISTORICAL_RETURNS)
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Portfolio value
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        pv = sum(1_000_000 * pos.quantity for pos in portfolio)
        
        # Simulate returns
        np.random.seed(123)
        simulated_returns = np.random.normal(mu, sigma, n_sims)
        simulated_pnl = simulated_returns * abs(pv)
        
        # Monte Carlo VaR
        mc_var = {}
        for alpha in confidence_levels:
            percentile = (1 - alpha) * 100
            mc_var[alpha] = -np.percentile(simulated_pnl, percentile)
        
        ctx.put(Keys.MONTE_CARLO_VAR, mc_var)
        
        if ctx.logger:
            for alpha, var in mc_var.items():
                ctx.logger.info("Monte Carlo VaR (%.0f%%): %.2f", alpha * 100, var)
        
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeExpectedShortfallStep(Step):
    """Step 7: Compute CVaR/ES for each method."""
    
    def run(self, ctx: Context) -> Context:
        var_cfg = _var_cfg(ctx.cfg)
        
        if not var_cfg.get("compute_es", True):
            return ctx
        
        returns = ctx.get(Keys.HISTORICAL_RETURNS)
        confidence_levels = var_cfg.get("confidence_levels", [0.95, 0.99])
        
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        pv = sum(1_000_000 * pos.quantity for pos in portfolio)
        
        # Historical ES
        expected_shortfall = {}
        for alpha in confidence_levels:
            percentile = (1 - alpha) * 100
            var_threshold = np.percentile(returns, percentile)
            tail_returns = returns[returns <= var_threshold]
            if len(tail_returns) > 0:
                es = -np.mean(tail_returns) * abs(pv)
            else:
                es = 0.0
            expected_shortfall[alpha] = es
        
        ctx.put(Keys.EXPECTED_SHORTFALL, {"historical": expected_shortfall})
        
        if ctx.logger:
            for alpha, es in expected_shortfall.items():
                ctx.logger.info("Expected Shortfall (%.0f%%): %.2f", alpha * 100, es)
        
        return ctx


@dataclass(frozen=True, slots=True)
class CompareVaRMethodsStep(Step):
    """Step 8: Compare results across methods."""
    
    def run(self, ctx: Context) -> Context:
        # Gather all VaR results
        comparison = {
            "historical": ctx.state.get(Keys.HISTORICAL_VAR, {}),
            "parametric": ctx.state.get(Keys.PARAMETRIC_VAR, {}),
            "monte_carlo": ctx.state.get(Keys.MONTE_CARLO_VAR, {}),
        }
        
        if ctx.logger:
            ctx.logger.info("VaR method comparison complete")
        
        return ctx


@dataclass(frozen=True, slots=True)
class WriteVaRReportStep(Step):
    """Step 9: Write VaR report."""
    
    def run(self, ctx: Context) -> Context:
        report = {
            "historical_var": ctx.state.get(Keys.HISTORICAL_VAR, {}),
            "parametric_var": ctx.state.get(Keys.PARAMETRIC_VAR, {}),
            "monte_carlo_var": ctx.state.get(Keys.MONTE_CARLO_VAR, {}),
            "expected_shortfall": ctx.state.get(Keys.EXPECTED_SHORTFALL, {}),
        }
        
        ctx.put(Keys.VAR_REPORT, report)
        
        if ctx.artifact_store:
            import json
            path = ctx.artifact_store.artifacts_root / "var_report.json"
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        
        if ctx.logger:
            ctx.logger.info("VaR report written")
        
        return ctx


# =============================================================================
# Pipeline Builder
# =============================================================================

def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the risk.compute_var pipeline."""
    return Pipeline(
        name="risk.compute_var",
        steps=[
            LoadPortfolioStep(name="load_portfolio"),
            LoadHistoricalDataStep(name="load_historical_data"),
            ComputeHistoricalVaRStep(name="compute_historical_var"),
            ComputeParametricVaRStep(name="compute_parametric_var"),
            ComputeMonteCarloVaRStep(name="compute_monte_carlo_var"),
            ComputeExpectedShortfallStep(name="compute_expected_shortfall"),
            CompareVaRMethodsStep(name="compare_var_methods"),
            WriteVaRReportStep(name="write_var_report"),
        ],
    )
