"""
Generic Portfolio Pricing Pipeline (V1)

This pipeline is intentionally minimal and stable. It assumes:
  - A Market snapshot exists in ctx.state["market"] (typically created by a marketdata pipeline)
  - A Portfolio exists in ctx.state["portfolio"] (positions carry position_id + quantity)

It then:
  1) Builds a PricerRegistry (V1: DefaultPricerRegistry)
  2) Prices the portfolio via PortfolioPricer
  3) Stores results in ctx.state for downstream reporting/artifacts

Design goals
------------
- Vn-proof: stable ctx.state keys, clear extension points.
- Generic: independent of FX/IR/Equity; relies on Portfolio + registry routing.
- Deterministic: registry construction is pure and repeatable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys

from src.portfolio.core import Portfolio
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry, PricerRegistry

# =============================================================================
# Config helpers (keep config parsing isolated + testable)
# =============================================================================

def _pricing_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """
    Extract the `pricing` config block from cfg.params.

    Notes
    -----
    - This pipeline only *requires* that cfg.params is a dict.
    - Pricing config is optional because we can price purely from ctx.state.
    """
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict.")
    block = cfg.params.get("pricing", {}) or {}
    if not isinstance(block, dict):
        raise TypeError("cfg.params['pricing'] must be a dict if provided.")
    return block


# =============================================================================
# Steps
# =============================================================================

@dataclass(frozen=True, slots=True)
class BuildPricerRegistryStep(Step):
    """
    Build the PricerRegistry used by PortfolioPricer.

    Outputs (stable keys)
    ---------------------
    ctx.state["pricer_registry"] : PricerRegistry

    Extension points (V2/Vn)
    ------------------------
    - Allow config-driven pricer overrides (named pricers, custom instruments).
    - Allow "registry.kind": "default" | "custom" | "hybrid".
    """

    def run(self, ctx: Context) -> Context:
        _ = _pricing_cfg(ctx.cfg)  # read config block (reserved for future Vn use)

        # Build the default registry (your existing, curated mapping).
        registry = DefaultPricerRegistry().build()

        # Store registry in state so downstream steps (and users) can inspect it.
        ctx.put(Keys.PRICER_REGISTRY, registry)

        # Log a small routing hint (keeps CLI output useful without prints).
        if ctx.logger is not None:
            ctx.logger.info("Built default PricerRegistry (%d default types).", len(registry.as_mapping()))

        return ctx


@dataclass(frozen=True, slots=True)
class PricePortfolioStep(Step):
    """
    Price a Portfolio using PortfolioPricer and the registry.

    Required inputs
    ---------------
    ctx.state["portfolio"] : Portfolio
    ctx.state["market"]    : Market
    ctx.state["pricer_registry"] : PricerRegistry

    Outputs (stable keys)
    ---------------------
    ctx.state["portfolio_pricing_result"]  : PortfolioPricer result object
    ctx.state["portfolio_pricing_summary"] : Dict[str, Any]
    """

    def run(self, ctx: Context) -> Context:
        # --- Pull required objects from state (fail fast with clear errors) ---
        if Keys.PORTFOLIO not in ctx.state:
            raise KeyError("Missing ctx.state['portfolio']. Provide a Portfolio before running pricing pipeline.")
        if Keys.MARKET not in ctx.state:
            raise KeyError("Missing ctx.state['market']. Run a marketdata snapshot step first (or inject Market).")
        if Keys.PRICER_REGISTRY not in ctx.state:
            raise KeyError("Missing ctx.state['pricer_registry']. BuildPricerRegistryStep did not run.")

        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        market = ctx.get(Keys.MARKET)
        registry: PricerRegistry = ctx.get(Keys.PRICER_REGISTRY)

        # --- Price using your portfolio pricer (this handles per-position PV + optional greeks) ---
        pricer = PortfolioPricer(pricer_registry=registry)
        result = pricer.price(portfolio, market)

        # --- Store full result for downstream reporting/artifacts ---
        ctx.put(Keys.PORTFOLIO_PRICING_RESULT, result)

        # --- Store a small summary that is JSON-friendly (good for logs and tests) ---
        summary: Dict[str, Any] = {
            "asof": getattr(market, "asof", None),
            "n_positions": len(getattr(portfolio, "positions", []) or []),
            "total_pv": float(result.totals.pv),
            "has_total_greeks": bool(result.totals.greeks),
        }
        ctx.put(Keys.PORTFOLIO_PRICING_SUMMARY, summary)

        # --- Log core outputs (no prints; prints happen in the example at the end) ---
        if ctx.logger is not None:
            ctx.logger.info(
                "PRICED_PORTFOLIO | asof=%s | n_positions=%d | total_pv=%.6f",
                summary["asof"],
                summary["n_positions"],
                summary["total_pv"],
            )

        return ctx


# =============================================================================
# Pipeline builder
# =============================================================================

def build_pipeline(cfg: RunConfig) -> Pipeline:
    """
    Build the built-in pricing pipeline.

    Notes
    -----
    - We accept cfg to keep the builder signature uniform across pipelines.
    - V1 ignores most pricing config because ctx.state carries Portfolio/Market.
    """
    _ = cfg  # keep signature uniform; avoids lint issues

    steps: List[Step] = [
        BuildPricerRegistryStep(name="build_pricer_registry"),
        PricePortfolioStep(name="price_portfolio"),
    ]

    return Pipeline(
        name="pricing.price_portfolio",
        steps=steps,
    )