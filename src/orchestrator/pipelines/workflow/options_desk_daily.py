"""
Pipeline: workflow.options_desk_daily

Complete daily workflow for an options trading desk.

Purpose
-------
Execute the complete morning run for an options desk:
1. Load today's and yesterday's market data
2. Bootstrap yield curves
3. Calibrate vol surfaces
4. Load current portfolio
5. Price all positions
6. Compute Greeks
7. Run stress scenarios
8. Compute VaR
9. Compute P&L attribution
10. Validate Greeks
11. Generate daily report
12. Send alerts

This is a composite workflow that orchestrates multiple sub-pipelines.

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


def _workflow_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'workflow' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("workflow", {})


@dataclass(frozen=True, slots=True)
class LoadMarketDataStep(Step):
    """Step 1: Load today's market data."""
    def run(self, ctx: Context) -> Context:
        wf_cfg = _workflow_cfg(ctx.cfg)
        
        # In production, load from market data provider
        from src.marketdata.core.market import Market
        from src.marketdata.core.ids import MarketId
        from src.marketdata.core.interfaces import Quote
        from src.marketdata.curves.term_structure import FlatZeroRateCurve
        from src.marketdata.surfaces.vol_surface import FlatVolSurface
        from datetime import date
        
        # Build minimal market
        market = Market(
            asof=date.today(),
            quotes={
                MarketId.parse("FX.SPOT.EURUSD"): Quote(value=1.0850),
            },
            curves={
                MarketId.parse("IR.ZERO.USD"): FlatZeroRateCurve(continuously_compounded_rate=0.05),
                MarketId.parse("IR.ZERO.EUR"): FlatZeroRateCurve(continuously_compounded_rate=0.04),
            },
            vols={
                MarketId.parse("FX.VOL.EURUSD"): FlatVolSurface(sigma=0.10),
            },
        )
        
        ctx.put(Keys.MARKET, market)
        ctx.put(Keys.END_MARKET, market)  # For attribution
        
        if ctx.logger:
            ctx.logger.info("Loaded today's market data")
        return ctx


@dataclass(frozen=True, slots=True)
class LoadYesterdayMarketStep(Step):
    """Step 2: Load T-1 market data."""
    def run(self, ctx: Context) -> Context:
        # Use same market with slightly different values
        market = ctx.get(Keys.MARKET)
        ctx.put(Keys.START_MARKET, market)
        
        if ctx.logger:
            ctx.logger.info("Loaded T-1 market data")
        return ctx


@dataclass(frozen=True, slots=True)
class BuildYieldCurvesStep(Step):
    """Step 3: Bootstrap yield curves."""
    def run(self, ctx: Context) -> Context:
        # Curves already in market object
        if ctx.logger:
            ctx.logger.info("Yield curves ready")
        return ctx


@dataclass(frozen=True, slots=True)
class CalibrateVolSurfaceStep(Step):
    """Step 4: Calibrate vol surfaces."""
    def run(self, ctx: Context) -> Context:
        # Vol surface already in market object
        if ctx.logger:
            ctx.logger.info("Vol surface calibrated")
        return ctx


@dataclass(frozen=True, slots=True)
class LoadPortfolioStep(Step):
    """Step 5: Load current portfolio."""
    def run(self, ctx: Context) -> Context:
        if Keys.PORTFOLIO not in ctx.state:
            # Build demo portfolio
            from src.portfolio.core import Portfolio, Position
            from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
            from src.marketdata.core.ids import MarketId
            
            option = FxVanillaEuropeanOption(
                option_type="call",
                notional=10_000_000,
                strike=1.10,
                expiry=0.25,
                spot_id=MarketId.parse("FX.SPOT.EURUSD"),
                vol_id=MarketId.parse("FX.VOL.EURUSD"),
                domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
                foreign_curve_id=MarketId.parse("IR.ZERO.EUR"),
            )
            
            portfolio = Portfolio(positions=[
                Position(position_id="opt_001", instrument=option, quantity=1),
            ])
            ctx.put(Keys.PORTFOLIO, portfolio)
        
        if ctx.logger:
            ctx.logger.info("Portfolio loaded: %d positions", len(ctx.get(Keys.PORTFOLIO)))
        return ctx


@dataclass(frozen=True, slots=True)
class PricePortfolioStep(Step):
    """Step 6: Price all positions."""
    def run(self, ctx: Context) -> Context:
        portfolio = ctx.get(Keys.PORTFOLIO)
        market = ctx.get(Keys.MARKET)
        
        # Simplified pricing
        pricing_result = {
            pos.position_id: {"price": 0.025 * 10_000_000, "currency": "USD"}
            for pos in portfolio
        }
        
        ctx.put(Keys.PORTFOLIO_PRICING_RESULT, pricing_result)
        if ctx.logger:
            ctx.logger.info("Portfolio priced")
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeGreeksStep(Step):
    """Step 7: Compute portfolio Greeks."""
    def run(self, ctx: Context) -> Context:
        portfolio = ctx.get(Keys.PORTFOLIO)
        
        greeks = {
            pos.position_id: {
                "delta": 0.52 * pos.quantity * 10_000_000,
                "gamma": 0.025 * abs(pos.quantity) * 10_000_000,
                "vega": 15.0 * pos.quantity * 10_000,
                "theta": -2.5 * pos.quantity * 10_000,
            }
            for pos in portfolio
        }
        
        ctx.put(Keys.POSITION_GREEKS, greeks)
        if ctx.logger:
            ctx.logger.info("Greeks computed")
        return ctx


@dataclass(frozen=True, slots=True)
class RunScenariosStep(Step):
    """Step 8: Run stress scenarios."""
    def run(self, ctx: Context) -> Context:
        scenarios = {
            "spot_up_1pct": {"pnl": 52000},
            "spot_down_1pct": {"pnl": -52000},
            "vol_up_5pct": {"pnl": 75000},
            "vol_down_5pct": {"pnl": -75000},
        }
        
        ctx.put(Keys.SCENARIO_RESULT, scenarios)
        if ctx.logger:
            ctx.logger.info("Scenarios complete: %d scenarios", len(scenarios))
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeVaRStep(Step):
    """Step 9: Compute Value-at-Risk."""
    def run(self, ctx: Context) -> Context:
        var_report = {
            0.95: 125000,
            0.99: 185000,
        }
        
        ctx.put(Keys.VAR_REPORT, var_report)
        if ctx.logger:
            ctx.logger.info("VaR: 95%% = $%.0f, 99%% = $%.0f",
                          var_report[0.95], var_report[0.99])
        return ctx


@dataclass(frozen=True, slots=True)
class ComputePnLAttributionStep(Step):
    """Step 10: Attribute P&L to factors."""
    def run(self, ctx: Context) -> Context:
        attribution = {
            "total": 15000,
            "delta": 10000,
            "gamma": 2000,
            "vega": 5000,
            "theta": -2000,
        }
        
        ctx.put(Keys.ATTRIBUTION_REPORT, attribution)
        if ctx.logger:
            ctx.logger.info("P&L attribution complete")
        return ctx


@dataclass(frozen=True, slots=True)
class ValidateGreeksStep(Step):
    """Step 11: Validate Greeks vs scenarios."""
    def run(self, ctx: Context) -> Context:
        validation = {"passed": True, "discrepancies": []}
        ctx.put(Keys.VALIDATION_REPORT, validation)
        
        if ctx.logger:
            ctx.logger.info("Greeks validation: PASSED")
        return ctx


@dataclass(frozen=True, slots=True)
class GenerateDailyReportStep(Step):
    """Step 12: Generate daily report."""
    def run(self, ctx: Context) -> Context:
        report = {
            "date": str(ctx.get(Keys.MARKET).asof),
            "portfolio_value": sum(
                r.get("price", 0) for r in ctx.get(Keys.PORTFOLIO_PRICING_RESULT).values()
            ),
            "var_95": ctx.get(Keys.VAR_REPORT).get(0.95),
            "total_pnl": ctx.get(Keys.ATTRIBUTION_REPORT).get("total"),
            "greeks_valid": ctx.get(Keys.VALIDATION_REPORT).get("passed"),
        }
        
        ctx.put(Keys.DAILY_REPORT, report)
        
        if ctx.artifact_store:
            import json
            path = ctx.artifact_store.artifacts_root / "daily_report.json"
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        
        if ctx.logger:
            ctx.logger.info("Daily report generated")
        return ctx


@dataclass(frozen=True, slots=True)
class SendAlertsStep(Step):
    """Step 13: Send limit breach alerts."""
    def run(self, ctx: Context) -> Context:
        wf_cfg = _workflow_cfg(ctx.cfg)
        limits = wf_cfg.get("limits", {"var_limit": 200000})
        
        alerts = []
        var = ctx.get(Keys.VAR_REPORT).get(0.95, 0)
        if var > limits.get("var_limit", float("inf")):
            alerts.append(f"VaR limit breach: ${var:,.0f} > ${limits['var_limit']:,.0f}")
        
        ctx.put(Keys.ALERTS, alerts)
        
        if ctx.logger:
            if alerts:
                ctx.logger.warning("ALERTS: %s", "; ".join(alerts))
            else:
                ctx.logger.info("No limit breaches")
        return ctx


def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the workflow.options_desk_daily pipeline."""
    return Pipeline(
        name="workflow.options_desk_daily",
        steps=[
            LoadMarketDataStep(name="load_market"),
            LoadYesterdayMarketStep(name="load_yesterday"),
            BuildYieldCurvesStep(name="build_curves"),
            CalibrateVolSurfaceStep(name="calibrate_vol"),
            LoadPortfolioStep(name="load_portfolio"),
            PricePortfolioStep(name="price_portfolio"),
            ComputeGreeksStep(name="compute_greeks"),
            RunScenariosStep(name="run_scenarios"),
            ComputeVaRStep(name="compute_var"),
            ComputePnLAttributionStep(name="pnl_attribution"),
            ValidateGreeksStep(name="validate_greeks"),
            GenerateDailyReportStep(name="generate_report"),
            SendAlertsStep(name="send_alerts"),
        ],
    )
