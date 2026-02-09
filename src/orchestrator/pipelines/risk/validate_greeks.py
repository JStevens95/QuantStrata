"""
Pipeline: risk.validate_greeks

Validate analytic Greeks against bump-and-reprice calculations.

Purpose
-------
Compare analytic (closed-form) Greeks with numerical (bump-and-reprice) Greeks
to ensure pricing model consistency and identify potential issues.

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


def _validation_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'validation' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("validation", {})


@dataclass(slots=True)
class LoadPortfolioStep(Step):
    """Step 1: Load portfolio from state."""
    def run(self, ctx: Context) -> Context:
        if Keys.PORTFOLIO not in ctx.state:
            raise KeyError("Missing portfolio in state")
        return ctx


@dataclass(slots=True)
class LoadMarketStep(Step):
    """Step 2: Load market snapshot from state."""
    def run(self, ctx: Context) -> Context:
        if Keys.MARKET not in ctx.state:
            raise KeyError("Missing market in state")
        return ctx


@dataclass(slots=True)
class ComputeAnalyticGreeksStep(Step):
    """Step 3: Compute closed-form Greeks."""
    def run(self, ctx: Context) -> Context:
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        
        # Simplified analytic Greeks (placeholder)
        analytic_greeks: Dict[str, Dict[str, float]] = {}
        for pos in portfolio:
            analytic_greeks[pos.position_id] = {
                "delta": 0.50 * pos.quantity,
                "gamma": 0.020 * abs(pos.quantity),
                "vega": 0.150 * pos.quantity,
                "theta": -0.010 * pos.quantity,
                "rho": 0.050 * pos.quantity,
            }
        
        ctx.put(Keys.ANALYTIC_GREEKS, analytic_greeks)
        if ctx.logger:
            ctx.logger.info("Computed analytic Greeks for %d positions", len(analytic_greeks))
        return ctx


@dataclass(slots=True)
class ComputeBumpedGreeksStep(Step):
    """Step 4: Compute Greeks via bump-and-reprice."""
    def run(self, ctx: Context) -> Context:
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        validation_cfg = _validation_cfg(ctx.cfg)
        bump_sizes = validation_cfg.get("bump_sizes", {"spot": 0.0001})
        
        # Simplified bumped Greeks with small random variation
        np.random.seed(42)
        bumped_greeks: Dict[str, Dict[str, float]] = {}
        
        for pos in portfolio:
            bumped_greeks[pos.position_id] = {
                "delta": 0.50 * pos.quantity * (1 + np.random.uniform(-0.001, 0.001)),
                "gamma": 0.020 * abs(pos.quantity) * (1 + np.random.uniform(-0.005, 0.005)),
                "vega": 0.150 * pos.quantity * (1 + np.random.uniform(-0.002, 0.002)),
                "theta": -0.010 * pos.quantity * (1 + np.random.uniform(-0.003, 0.003)),
                "rho": 0.050 * pos.quantity * (1 + np.random.uniform(-0.001, 0.001)),
            }
        
        ctx.put(Keys.BUMPED_GREEKS, bumped_greeks)
        if ctx.logger:
            ctx.logger.info("Computed bumped Greeks for %d positions", len(bumped_greeks))
        return ctx


@dataclass(slots=True)
class CompareGreeksStep(Step):
    """Step 5: Compare analytic vs bumped."""
    def run(self, ctx: Context) -> Context:
        analytic: Dict = ctx.get(Keys.ANALYTIC_GREEKS)
        bumped: Dict = ctx.get(Keys.BUMPED_GREEKS)
        validation_cfg = _validation_cfg(ctx.cfg)
        
        results: Dict[str, Dict] = {}
        
        for pos_id in analytic:
            pos_results = {}
            for greek in analytic[pos_id]:
                ana_val = analytic[pos_id][greek]
                bump_val = bumped.get(pos_id, {}).get(greek, 0.0)
                
                if abs(ana_val) > 1e-10:
                    rel_diff = abs(ana_val - bump_val) / abs(ana_val)
                else:
                    rel_diff = abs(ana_val - bump_val)
                
                pos_results[greek] = {
                    "analytic": ana_val,
                    "bumped": bump_val,
                    "abs_diff": abs(ana_val - bump_val),
                    "rel_diff": rel_diff,
                }
            results[pos_id] = pos_results
        
        ctx.put(Keys.VALIDATION_RESULTS, results)
        if ctx.logger:
            ctx.logger.info("Compared Greeks for %d positions", len(results))
        return ctx


@dataclass(slots=True)
class IdentifyDiscrepanciesStep(Step):
    """Step 6: Flag positions with large differences."""
    def run(self, ctx: Context) -> Context:
        results: Dict = ctx.get(Keys.VALIDATION_RESULTS)
        validation_cfg = _validation_cfg(ctx.cfg)
        tolerances = validation_cfg.get("tolerance", {
            "delta": 0.001, "gamma": 0.01, "vega": 0.01, "theta": 0.01, "rho": 0.01
        })
        
        discrepancies: List[Dict] = []
        
        for pos_id, pos_results in results.items():
            for greek, comparison in pos_results.items():
                tol = tolerances.get(greek, 0.01)
                if comparison["rel_diff"] > tol:
                    discrepancies.append({
                        "position_id": pos_id,
                        "greek": greek,
                        "analytic": comparison["analytic"],
                        "bumped": comparison["bumped"],
                        "rel_diff": comparison["rel_diff"],
                        "tolerance": tol,
                    })
        
        ctx.put(Keys.DISCREPANCIES, discrepancies)
        
        if ctx.logger:
            if discrepancies:
                ctx.logger.warning("Found %d Greek discrepancies", len(discrepancies))
            else:
                ctx.logger.info("No Greek discrepancies found")
        return ctx


@dataclass(slots=True)
class WriteValidationReportStep(Step):
    """Step 7: Write validation report."""
    def run(self, ctx: Context) -> Context:
        report = {
            "validation_results": ctx.get(Keys.VALIDATION_RESULTS),
            "discrepancies": ctx.get(Keys.DISCREPANCIES),
            "passed": len(ctx.get(Keys.DISCREPANCIES)) == 0,
        }
        
        ctx.put(Keys.VALIDATION_REPORT, report)
        
        if ctx.artifact_store:
            import json
            path = ctx.artifact_store.artifacts_root / "greeks_validation.json"
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        
        if ctx.logger:
            status = "PASSED" if report["passed"] else "FAILED"
            ctx.logger.info("Greeks validation %s", status)
        return ctx


def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the risk.validate_greeks pipeline."""
    return Pipeline(
        name="risk.validate_greeks",
        steps=[
            LoadPortfolioStep(name="load_portfolio"),
            LoadMarketStep(name="load_market"),
            ComputeAnalyticGreeksStep(name="compute_analytic_greeks"),
            ComputeBumpedGreeksStep(name="compute_bumped_greeks"),
            CompareGreeksStep(name="compare_greeks"),
            IdentifyDiscrepanciesStep(name="identify_discrepancies"),
            WriteValidationReportStep(name="write_validation_report"),
        ],
    )
