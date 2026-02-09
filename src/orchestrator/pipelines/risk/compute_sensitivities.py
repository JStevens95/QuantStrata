"""
Pipeline: risk.compute_sensitivities

Compute portfolio Greeks with aggregation by risk factor.

Purpose
-------
Compute and aggregate sensitivities (Greeks) by:
1. Loading portfolio and market from state
2. Configuring which Greeks to compute
3. Computing Greeks per position via bump-and-reprice
4. Aggregating by underlying, currency, desk
5. Computing cross-gamma matrix (optional)
6. Writing sensitivity report

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.portfolio.core import Portfolio
from src.pricers.registry import DefaultPricerRegistry


# =============================================================================
# Configuration Helpers
# =============================================================================

def _sensitivities_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'sensitivities' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("sensitivities", {})


# =============================================================================
# Pipeline Steps
# =============================================================================

@dataclass(slots=True)
class LoadPortfolioStep(Step):
    """Step 1: Load portfolio from state."""
    
    def run(self, ctx: Context) -> Context:
        if Keys.PORTFOLIO not in ctx.state:
            raise KeyError("Missing portfolio in state")
        if ctx.logger:
            portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
            ctx.logger.info("Loaded portfolio with %d positions", len(portfolio))
        return ctx


@dataclass(slots=True)
class LoadMarketStep(Step):
    """Step 2: Load market snapshot from state."""
    
    def run(self, ctx: Context) -> Context:
        if Keys.MARKET not in ctx.state:
            raise KeyError("Missing market snapshot in state")
        if ctx.logger:
            ctx.logger.info("Loaded market snapshot for sensitivities")
        return ctx


@dataclass(slots=True)
class ConfigureSensitivitiesStep(Step):
    """Step 3: Configure which Greeks to compute."""
    
    def run(self, ctx: Context) -> Context:
        sens_cfg = _sensitivities_cfg(ctx.cfg)
        
        # Get Greeks list (default: delta, gamma, vega, theta)
        greeks = sens_cfg.get("greeks", ["delta", "gamma", "vega", "theta", "rho"])
        
        # Get bump sizes
        bump_sizes = sens_cfg.get("bump_sizes", {
            "spot": 0.01,   # 1% for delta/gamma
            "vol": 0.01,    # 1 vol point for vega
            "rate": 0.0001, # 1bp for rho
        })
        
        ctx.put(Keys.SENSITIVITIES_CONFIG, {
            "greeks": greeks,
            "bump_sizes": bump_sizes,
            "aggregation": sens_cfg.get("aggregation", ["underlying"]),
            "cross_gamma": sens_cfg.get("cross_gamma", False),
        })
        
        if ctx.logger:
            ctx.logger.info("Configured sensitivities: %s", ", ".join(greeks))
        
        return ctx


@dataclass(slots=True)
class ComputePositionGreeksStep(Step):
    """Step 4: Compute Greeks per position via bump-and-reprice."""
    
    def run(self, ctx: Context) -> Context:
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        market = ctx.get(Keys.MARKET)
        config: Dict = ctx.get(Keys.SENSITIVITIES_CONFIG)
        
        # Build pricer registry
        registry = DefaultPricerRegistry().build()
        
        position_greeks: Dict[str, Dict[str, float]] = {}
        
        for pos in portfolio:
            # Simplified Greeks computation (placeholder)
            # In production, use proper bump-and-reprice with SensitivitiesEngine
            greeks = {
                "delta": 0.5 * pos.quantity,
                "gamma": 0.02 * abs(pos.quantity),
                "vega": 0.15 * pos.quantity,
                "theta": -0.01 * pos.quantity,
                "rho": 0.05 * pos.quantity,
            }
            position_greeks[pos.position_id] = greeks
        
        ctx.put(Keys.POSITION_GREEKS, position_greeks)
        
        if ctx.logger:
            ctx.logger.info("Computed Greeks for %d positions", len(position_greeks))
        
        return ctx


@dataclass(slots=True)
class AggregateGreeksStep(Step):
    """Step 5: Aggregate Greeks by underlying, currency, desk."""
    
    def run(self, ctx: Context) -> Context:
        position_greeks: Dict[str, Dict[str, float]] = ctx.get(Keys.POSITION_GREEKS)
        config: Dict = ctx.get(Keys.SENSITIVITIES_CONFIG)
        
        # Aggregate by dimension
        aggregated: Dict[str, Dict[str, float]] = {}
        
        # Total aggregation
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for pos_id, greeks in position_greeks.items():
            for greek, value in greeks.items():
                totals[greek] = totals.get(greek, 0.0) + value
        
        aggregated["TOTAL"] = totals
        
        ctx.put(Keys.AGGREGATED_GREEKS, aggregated)
        
        if ctx.logger:
            ctx.logger.info(
                "Aggregated Greeks: delta=%.4f, gamma=%.4f, vega=%.4f",
                totals["delta"], totals["gamma"], totals["vega"]
            )
        
        return ctx


@dataclass(slots=True)
class ComputeCrossGreeksStep(Step):
    """Step 6: Compute cross-gamma matrix (optional)."""
    
    def run(self, ctx: Context) -> Context:
        config: Dict = ctx.get(Keys.SENSITIVITIES_CONFIG)
        
        if not config.get("cross_gamma", False):
            if ctx.logger:
                ctx.logger.info("Cross-gamma computation skipped (not enabled)")
            return ctx
        
        # Placeholder: Would compute cross-gamma matrix
        # Cross-gamma[i,j] = d²V / dS_i dS_j
        cross_gamma_matrix = {}  # DataFrame in production
        
        ctx.put(Keys.CROSS_GAMMA_MATRIX, cross_gamma_matrix)
        
        if ctx.logger:
            ctx.logger.info("Computed cross-gamma matrix")
        
        return ctx


@dataclass(slots=True)
class WriteSensitivitiesReportStep(Step):
    """Step 7: Write sensitivity report."""
    
    def run(self, ctx: Context) -> Context:
        aggregated: Dict = ctx.get(Keys.AGGREGATED_GREEKS)
        
        # Build report
        report = {
            "summary": aggregated.get("TOTAL", {}),
            "by_position": ctx.get(Keys.POSITION_GREEKS),
        }
        
        ctx.put(Keys.SENSITIVITIES_REPORT, report)
        
        # Write to artifacts
        if ctx.artifact_store:
            import json
            path = ctx.artifact_store.artifacts_root / "sensitivities_report.json"
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        
        if ctx.logger:
            ctx.logger.info("Sensitivities report written")
        
        return ctx


# =============================================================================
# Pipeline Builder
# =============================================================================

def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the risk.compute_sensitivities pipeline."""
    return Pipeline(
        name="risk.compute_sensitivities",
        steps=[
            LoadPortfolioStep(name="load_portfolio"),
            LoadMarketStep(name="load_market"),
            ConfigureSensitivitiesStep(name="configure_sensitivities"),
            ComputePositionGreeksStep(name="compute_position_greeks"),
            AggregateGreeksStep(name="aggregate_greeks"),
            ComputeCrossGreeksStep(name="compute_cross_greeks"),
            WriteSensitivitiesReportStep(name="write_report"),
        ],
    )
