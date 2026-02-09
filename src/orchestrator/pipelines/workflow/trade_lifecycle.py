"""
Pipeline: workflow.trade_lifecycle

Trade lifecycle management from request to settlement.

Purpose
-------
Manage the complete lifecycle of a new trade:
1. Receive trade request
2. Validate trade parameters
3. Price the trade
4. Check risk limits
5. Book the trade
6. Update portfolio
7. Compute incremental Greeks
8. Generate booking confirmation
9. Update risk reports

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys


def _lifecycle_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'trade_lifecycle' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("trade_lifecycle", {})


@dataclass(slots=True)
class ReceiveTradeRequestStep(Step):
    """Step 1: Receive trade request."""
    def run(self, ctx: Context) -> Context:
        lc_cfg = _lifecycle_cfg(ctx.cfg)
        
        trade_request = lc_cfg.get("trade_request", {
            "instrument_type": "FxVanillaOption",
            "underlying": "EURUSD",
            "strike": 1.10,
            "expiry": "3M",
            "option_type": "call",
            "notional": 10_000_000,
            "direction": "buy",
        })
        
        ctx.put(Keys.TRADE_REQUEST, trade_request)
        if ctx.logger:
            ctx.logger.info("Received trade request: %s %s",
                          trade_request.get("direction"),
                          trade_request.get("instrument_type"))
        return ctx


@dataclass(slots=True)
class ValidateTradeStep(Step):
    """Step 2: Validate trade parameters."""
    def run(self, ctx: Context) -> Context:
        trade = ctx.get(Keys.TRADE_REQUEST)
        
        errors = []
        if trade.get("notional", 0) <= 0:
            errors.append("Notional must be positive")
        if trade.get("strike", 0) <= 0:
            errors.append("Strike must be positive")
        
        validation = {
            "valid": len(errors) == 0,
            "errors": errors,
        }
        
        ctx.put(Keys.TRADE_VALIDATION, validation)
        if ctx.logger:
            status = "VALID" if validation["valid"] else "INVALID"
            ctx.logger.info("Trade validation: %s", status)
        return ctx


@dataclass(slots=True)
class PriceTradeStep(Step):
    """Step 3: Price the trade."""
    def run(self, ctx: Context) -> Context:
        trade = ctx.get(Keys.TRADE_REQUEST)
        
        # Simplified pricing
        price_pct = 0.025  # 2.5% of notional
        price = trade.get("notional", 0) * price_pct
        
        pricing = {
            "mid_price": price,
            "bid_price": price * 0.98,
            "ask_price": price * 1.02,
            "delta": 0.52 * trade.get("notional", 0),
            "vega": 0.15 * trade.get("notional", 0) / 100,
        }
        
        ctx.put("trade_pricing", pricing)
        if ctx.logger:
            ctx.logger.info("Trade price: $%.0f (mid)", pricing["mid_price"])
        return ctx


@dataclass(slots=True)
class CheckRiskLimitsStep(Step):
    """Step 4: Check risk limits."""
    def run(self, ctx: Context) -> Context:
        pricing = ctx.state.get("trade_pricing", {})
        lc_cfg = _lifecycle_cfg(ctx.cfg)
        limits = lc_cfg.get("limits", {
            "max_notional": 50_000_000,
            "max_delta": 100_000_000,
        })
        
        trade = ctx.get(Keys.TRADE_REQUEST)
        
        breaches = []
        if trade.get("notional", 0) > limits.get("max_notional", float("inf")):
            breaches.append("Max notional exceeded")
        if abs(pricing.get("delta", 0)) > limits.get("max_delta", float("inf")):
            breaches.append("Max delta exceeded")
        
        ctx.put("limit_check", {
            "passed": len(breaches) == 0,
            "breaches": breaches,
        })
        
        if ctx.logger:
            if breaches:
                ctx.logger.warning("Limit breaches: %s", ", ".join(breaches))
            else:
                ctx.logger.info("Risk limits: PASSED")
        return ctx


@dataclass(slots=True)
class BookTradeStep(Step):
    """Step 5: Book the trade."""
    def run(self, ctx: Context) -> Context:
        validation = ctx.get(Keys.TRADE_VALIDATION)
        limit_check = ctx.state.get("limit_check", {"passed": True})
        
        if not validation.get("valid") or not limit_check.get("passed"):
            ctx.put(Keys.TRADE_BOOKING, {"booked": False, "reason": "Validation/limits failed"})
            return ctx
        
        # Generate booking ID
        import uuid
        booking_id = f"TRD-{uuid.uuid4().hex[:8].upper()}"
        
        ctx.put(Keys.TRADE_BOOKING, {
            "booked": True,
            "booking_id": booking_id,
            "timestamp": str(__import__("datetime").datetime.now()),
        })
        
        if ctx.logger:
            ctx.logger.info("Trade booked: %s", booking_id)
        return ctx


@dataclass(slots=True)
class UpdatePortfolioStep(Step):
    """Step 6: Update portfolio."""
    def run(self, ctx: Context) -> Context:
        booking = ctx.get(Keys.TRADE_BOOKING)
        
        if not booking.get("booked"):
            return ctx
        
        # In production, add trade to portfolio
        if ctx.logger:
            ctx.logger.info("Portfolio updated with new trade")
        return ctx


@dataclass(slots=True)
class ComputeIncrementalGreeksStep(Step):
    """Step 7: Compute incremental Greeks."""
    def run(self, ctx: Context) -> Context:
        pricing = ctx.state.get("trade_pricing", {})
        
        incremental_greeks = {
            "delta": pricing.get("delta", 0),
            "vega": pricing.get("vega", 0),
        }
        
        ctx.put("incremental_greeks", incremental_greeks)
        if ctx.logger:
            ctx.logger.info("Incremental Greeks: delta=$%.0f", incremental_greeks["delta"])
        return ctx


@dataclass(slots=True)
class GenerateConfirmationStep(Step):
    """Step 8: Generate booking confirmation."""
    def run(self, ctx: Context) -> Context:
        booking = ctx.get(Keys.TRADE_BOOKING)
        trade = ctx.get(Keys.TRADE_REQUEST)
        pricing = ctx.state.get("trade_pricing", {})
        
        confirmation = {
            "booking_id": booking.get("booking_id"),
            "trade_details": trade,
            "pricing": pricing,
            "status": "CONFIRMED" if booking.get("booked") else "REJECTED",
        }
        
        ctx.put("confirmation", confirmation)
        
        if ctx.artifact_store:
            import json
            path = ctx.artifact_store.artifacts_root / f"confirmation_{booking.get('booking_id', 'none')}.json"
            with open(path, "w") as f:
                json.dump(confirmation, f, indent=2, default=str)
        return ctx


@dataclass(slots=True)
class UpdateRiskReportsStep(Step):
    """Step 9: Update risk reports."""
    def run(self, ctx: Context) -> Context:
        report = {
            "trade": ctx.get(Keys.TRADE_BOOKING),
            "validation": ctx.get(Keys.TRADE_VALIDATION),
            "pricing": ctx.state.get("trade_pricing"),
            "incremental_greeks": ctx.state.get("incremental_greeks"),
        }
        
        ctx.put(Keys.TRADE_LIFECYCLE_REPORT, report)
        
        if ctx.logger:
            ctx.logger.info("Trade lifecycle complete")
        return ctx


def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the workflow.trade_lifecycle pipeline."""
    return Pipeline(
        name="workflow.trade_lifecycle",
        steps=[
            ReceiveTradeRequestStep(name="receive_request"),
            ValidateTradeStep(name="validate_trade"),
            PriceTradeStep(name="price_trade"),
            CheckRiskLimitsStep(name="check_limits"),
            BookTradeStep(name="book_trade"),
            UpdatePortfolioStep(name="update_portfolio"),
            ComputeIncrementalGreeksStep(name="compute_greeks"),
            GenerateConfirmationStep(name="generate_confirmation"),
            UpdateRiskReportsStep(name="update_reports"),
        ],
    )
