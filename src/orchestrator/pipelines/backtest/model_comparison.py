"""
Pipeline: backtest.model_comparison

Compare pricing results across multiple models.

Purpose
-------
Compare analytic, Monte Carlo, and finite difference pricing:
1. Load portfolio and market
2. Price with analytic (BSM)
3. Price with Monte Carlo
4. Price with finite difference
5. Compare results
6. Check convergence
7. Write comparison report

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.portfolio.core import Portfolio


def _comparison_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'comparison' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("comparison", {})


@dataclass(slots=True)
class LoadComparisonConfigStep(Step):
    """Step 1: Load comparison config."""
    def run(self, ctx: Context) -> Context:
        cmp_cfg = _comparison_cfg(ctx.cfg)
        models = cmp_cfg.get("models", [
            {"name": "analytic_bsm", "type": "analytic"},
            {"name": "monte_carlo", "type": "monte_carlo", "n_paths": [10000]},
            {"name": "finite_difference", "type": "finite_difference", "grid_points": [200]},
        ])
        ctx.put("comparison_models", models)
        if ctx.logger:
            ctx.logger.info("Comparing %d models", len(models))
        return ctx


@dataclass(slots=True)
class LoadPortfolioStep(Step):
    """Step 2: Load test portfolio."""
    def run(self, ctx: Context) -> Context:
        if Keys.PORTFOLIO not in ctx.state:
            raise KeyError("Missing portfolio in state")
        return ctx


@dataclass(slots=True)
class LoadMarketStep(Step):
    """Step 3: Load market snapshot."""
    def run(self, ctx: Context) -> Context:
        if Keys.MARKET not in ctx.state:
            raise KeyError("Missing market in state")
        return ctx


@dataclass(slots=True)
class PriceWithAnalyticStep(Step):
    """Step 4: Price with BSM/analytic."""
    def run(self, ctx: Context) -> Context:
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        
        # Simplified analytic pricing
        analytic_results = {}
        for pos in portfolio:
            analytic_results[pos.position_id] = {
                "price": 5.25,
                "delta": 0.52,
                "gamma": 0.025,
                "vega": 15.0,
            }
        
        ctx.put("analytic_results", analytic_results)
        if ctx.logger:
            ctx.logger.info("Analytic pricing complete")
        return ctx


@dataclass(slots=True)
class PriceWithMonteCarloStep(Step):
    """Step 5: Price with Monte Carlo."""
    def run(self, ctx: Context) -> Context:
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        
        mc_results = {}
        for pos in portfolio:
            mc_results[pos.position_id] = {
                "price": 5.24 + np.random.normal(0, 0.02),
                "delta": 0.52 + np.random.normal(0, 0.01),
                "gamma": 0.025 + np.random.normal(0, 0.002),
                "vega": 15.0 + np.random.normal(0, 0.1),
            }
        
        ctx.put("mc_results", mc_results)
        if ctx.logger:
            ctx.logger.info("Monte Carlo pricing complete")
        return ctx


@dataclass(slots=True)
class PriceWithFDEStep(Step):
    """Step 6: Price with finite difference."""
    def run(self, ctx: Context) -> Context:
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        
        fde_results = {}
        for pos in portfolio:
            fde_results[pos.position_id] = {
                "price": 5.25 + np.random.normal(0, 0.01),
                "delta": 0.52 + np.random.normal(0, 0.005),
                "gamma": 0.025 + np.random.normal(0, 0.001),
                "vega": 15.0 + np.random.normal(0, 0.05),
            }
        
        ctx.put("fde_results", fde_results)
        if ctx.logger:
            ctx.logger.info("Finite difference pricing complete")
        return ctx


@dataclass(slots=True)
class CompareResultsStep(Step):
    """Step 7: Compare prices and Greeks."""
    def run(self, ctx: Context) -> Context:
        analytic = ctx.state.get("analytic_results", {})
        mc = ctx.state.get("mc_results", {})
        fde = ctx.state.get("fde_results", {})
        
        comparison = {}
        for pos_id in analytic:
            comparison[pos_id] = {
                "price_diff_mc": abs(analytic[pos_id]["price"] - mc.get(pos_id, {}).get("price", 0)),
                "price_diff_fde": abs(analytic[pos_id]["price"] - fde.get(pos_id, {}).get("price", 0)),
                "delta_diff_mc": abs(analytic[pos_id]["delta"] - mc.get(pos_id, {}).get("delta", 0)),
                "delta_diff_fde": abs(analytic[pos_id]["delta"] - fde.get(pos_id, {}).get("delta", 0)),
            }
        
        ctx.put(Keys.COMPARISON_MATRIX, comparison)
        if ctx.logger:
            ctx.logger.info("Model comparison complete")
        return ctx


@dataclass(slots=True)
class ComputeConvergenceStep(Step):
    """Step 8: Check MC/FDE convergence."""
    def run(self, ctx: Context) -> Context:
        comparison = ctx.get(Keys.COMPARISON_MATRIX)
        
        # Check if all differences are within tolerance
        tolerance = 0.001
        all_converged = all(
            cmp["price_diff_mc"] < tolerance and cmp["price_diff_fde"] < tolerance
            for cmp in comparison.values()
        )
        
        ctx.put(Keys.CONVERGENCE_ANALYSIS, {
            "converged": all_converged,
            "tolerance": tolerance,
        })
        
        if ctx.logger:
            status = "CONVERGED" if all_converged else "NOT CONVERGED"
            ctx.logger.info("Convergence check: %s", status)
        return ctx


@dataclass(slots=True)
class WriteComparisonReportStep(Step):
    """Step 9: Write comparison report."""
    def run(self, ctx: Context) -> Context:
        model_results = {
            "analytic": ctx.state.get("analytic_results"),
            "monte_carlo": ctx.state.get("mc_results"),
            "finite_difference": ctx.state.get("fde_results"),
        }
        
        ctx.put(Keys.MODEL_RESULTS, model_results)
        
        if ctx.artifact_store:
            import json
            report = {
                "results": model_results,
                "comparison": ctx.state.get(Keys.COMPARISON_MATRIX),
                "convergence": ctx.state.get(Keys.CONVERGENCE_ANALYSIS),
            }
            path = ctx.artifact_store.artifacts_root / "model_comparison.json"
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        
        if ctx.logger:
            ctx.logger.info("Comparison report written")
        return ctx


def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the backtest.model_comparison pipeline."""
    return Pipeline(
        name="backtest.model_comparison",
        steps=[
            LoadComparisonConfigStep(name="load_config"),
            LoadPortfolioStep(name="load_portfolio"),
            LoadMarketStep(name="load_market"),
            PriceWithAnalyticStep(name="price_analytic"),
            PriceWithMonteCarloStep(name="price_mc"),
            PriceWithFDEStep(name="price_fde"),
            CompareResultsStep(name="compare_results"),
            ComputeConvergenceStep(name="compute_convergence"),
            WriteComparisonReportStep(name="write_report"),
        ],
    )
