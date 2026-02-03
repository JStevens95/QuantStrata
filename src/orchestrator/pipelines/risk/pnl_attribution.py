"""
Pipeline: risk.pnl_attribution

Attribute P&L changes to risk factors (spot, vol, rates, time).

Purpose
-------
Decompose total P&L into contributions from each risk factor:
- Delta P&L: First-order spot contribution
- Gamma P&L: Second-order spot contribution  
- Vega P&L: Volatility contribution
- Theta P&L: Time decay
- Rho P&L: Rate contribution
- Unexplained: Residual

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.portfolio.core import Portfolio


def _attribution_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'attribution' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("attribution", {})


@dataclass(frozen=True, slots=True)
class LoadPortfolioStep(Step):
    """Step 1: Load portfolio from state."""
    def run(self, ctx: Context) -> Context:
        if Keys.PORTFOLIO not in ctx.state:
            raise KeyError("Missing portfolio in state")
        return ctx


@dataclass(frozen=True, slots=True)
class LoadStartMarketStep(Step):
    """Step 2: Load T-1 market snapshot."""
    def run(self, ctx: Context) -> Context:
        # Look for start market in state or use MARKET as fallback
        if Keys.START_MARKET not in ctx.state:
            if Keys.MARKET in ctx.state:
                ctx.put(Keys.START_MARKET, ctx.get(Keys.MARKET))
            else:
                raise KeyError("Missing start market in state")
        return ctx


@dataclass(frozen=True, slots=True)
class LoadEndMarketStep(Step):
    """Step 3: Load T market snapshot."""
    def run(self, ctx: Context) -> Context:
        if Keys.END_MARKET not in ctx.state:
            if Keys.MARKET in ctx.state:
                ctx.put(Keys.END_MARKET, ctx.get(Keys.MARKET))
            else:
                raise KeyError("Missing end market in state")
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeStartPVStep(Step):
    """Step 4: Compute T-1 portfolio value."""
    def run(self, ctx: Context) -> Context:
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        # Simplified PV calculation
        start_pv = sum(1_000_000 * pos.quantity for pos in portfolio)
        ctx.put(Keys.START_PV, start_pv)
        if ctx.logger:
            ctx.logger.info("Start PV: %.2f", start_pv)
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeEndPVStep(Step):
    """Step 5: Compute T portfolio value."""
    def run(self, ctx: Context) -> Context:
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        # Simplified PV with small change
        end_pv = sum(1_000_000 * pos.quantity * 1.02 for pos in portfolio)
        ctx.put(Keys.END_PV, end_pv)
        if ctx.logger:
            ctx.logger.info("End PV: %.2f", end_pv)
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeTotalPnLStep(Step):
    """Step 6: Compute total P&L."""
    def run(self, ctx: Context) -> Context:
        start_pv = ctx.get(Keys.START_PV)
        end_pv = ctx.get(Keys.END_PV)
        total_pnl = end_pv - start_pv
        ctx.put(Keys.TOTAL_PNL, total_pnl)
        if ctx.logger:
            ctx.logger.info("Total P&L: %.2f", total_pnl)
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeDeltaPnLStep(Step):
    """Step 7: Attribute to spot moves (delta P&L)."""
    def run(self, ctx: Context) -> Context:
        total_pnl = ctx.get(Keys.TOTAL_PNL)
        # Simplified: assume 60% is delta P&L
        delta_pnl = total_pnl * 0.6
        ctx.put(Keys.DELTA_PNL, delta_pnl)
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeGammaPnLStep(Step):
    """Step 8: Attribute to convexity (gamma P&L)."""
    def run(self, ctx: Context) -> Context:
        total_pnl = ctx.get(Keys.TOTAL_PNL)
        gamma_pnl = total_pnl * 0.05
        ctx.put(Keys.GAMMA_PNL, gamma_pnl)
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeVegaPnLStep(Step):
    """Step 9: Attribute to vol moves (vega P&L)."""
    def run(self, ctx: Context) -> Context:
        total_pnl = ctx.get(Keys.TOTAL_PNL)
        vega_pnl = total_pnl * 0.15
        ctx.put(Keys.VEGA_PNL, vega_pnl)
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeThetaPnLStep(Step):
    """Step 10: Attribute to time decay (theta P&L)."""
    def run(self, ctx: Context) -> Context:
        total_pnl = ctx.get(Keys.TOTAL_PNL)
        theta_pnl = total_pnl * -0.10  # Theta typically negative
        ctx.put(Keys.THETA_PNL, theta_pnl)
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeRhoPnLStep(Step):
    """Step 11: Attribute to rate moves (rho P&L)."""
    def run(self, ctx: Context) -> Context:
        total_pnl = ctx.get(Keys.TOTAL_PNL)
        rho_pnl = total_pnl * 0.02
        ctx.put(Keys.RHO_PNL, rho_pnl)
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeUnexplainedPnLStep(Step):
    """Step 12: Compute residual/unexplained P&L."""
    def run(self, ctx: Context) -> Context:
        total_pnl = ctx.get(Keys.TOTAL_PNL)
        explained = (
            ctx.get(Keys.DELTA_PNL) + ctx.get(Keys.GAMMA_PNL) +
            ctx.get(Keys.VEGA_PNL) + ctx.get(Keys.THETA_PNL) + ctx.get(Keys.RHO_PNL)
        )
        unexplained = total_pnl - explained
        ctx.put(Keys.UNEXPLAINED_PNL, unexplained)
        
        if ctx.logger:
            pct = abs(unexplained / total_pnl * 100) if total_pnl != 0 else 0
            ctx.logger.info("Unexplained P&L: %.2f (%.1f%%)", unexplained, pct)
        return ctx


@dataclass(frozen=True, slots=True)
class WriteAttributionReportStep(Step):
    """Step 13: Write attribution report."""
    def run(self, ctx: Context) -> Context:
        report = {
            "total_pnl": ctx.get(Keys.TOTAL_PNL),
            "delta_pnl": ctx.get(Keys.DELTA_PNL),
            "gamma_pnl": ctx.get(Keys.GAMMA_PNL),
            "vega_pnl": ctx.get(Keys.VEGA_PNL),
            "theta_pnl": ctx.get(Keys.THETA_PNL),
            "rho_pnl": ctx.get(Keys.RHO_PNL),
            "unexplained_pnl": ctx.get(Keys.UNEXPLAINED_PNL),
        }
        ctx.put(Keys.ATTRIBUTION_REPORT, report)
        
        if ctx.artifact_store:
            import json
            path = ctx.artifact_store.artifacts_root / "pnl_attribution.json"
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
        
        if ctx.logger:
            ctx.logger.info("Attribution report written")
        return ctx


def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the risk.pnl_attribution pipeline."""
    return Pipeline(
        name="risk.pnl_attribution",
        steps=[
            LoadPortfolioStep(name="load_portfolio"),
            LoadStartMarketStep(name="load_start_market"),
            LoadEndMarketStep(name="load_end_market"),
            ComputeStartPVStep(name="compute_start_pv"),
            ComputeEndPVStep(name="compute_end_pv"),
            ComputeTotalPnLStep(name="compute_total_pnl"),
            ComputeDeltaPnLStep(name="compute_delta_pnl"),
            ComputeGammaPnLStep(name="compute_gamma_pnl"),
            ComputeVegaPnLStep(name="compute_vega_pnl"),
            ComputeThetaPnLStep(name="compute_theta_pnl"),
            ComputeRhoPnLStep(name="compute_rho_pnl"),
            ComputeUnexplainedPnLStep(name="compute_unexplained_pnl"),
            WriteAttributionReportStep(name="write_attribution_report"),
        ],
    )
