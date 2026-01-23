"""
Example: Marketdata (timeseries) -> Snapshot -> Price FX Portfolio -> Spot sweep plots

This script demonstrates an end-to-end workflow suitable for a front-office
quant library "examples" folder:

1) Build an in-memory MarketDataset + a Market snapshot using a built-in pipeline.
2) Build a Position-based Portfolio (Position owns identity + quantity).
3) Price the portfolio by running the *pricing pipeline* (not direct PortfolioPricer calls).
4) Run a spot sweep (PV / PnL / Greeks vs spot) WITHOUT mutating Market internals.
5) Generate clean, report-friendly plots (PNG + PDF optional).

Design goals
------------
- Config uses canonical MarketId strings only (MarketId.parse compatible).
- Pipeline logging happens during execution; plain prints happen at the end.
- Marketdata + pricing both run through PipelineRunner (so both are logged).
- Sweeps use a MarketOverride wrapper (clean, avoids quote dict mutation bugs).
- Matplotlib-only plots with consistent styling (CI-friendly, dependency-light).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Mapping

import numpy as np

# ---------------- Orchestrator imports ----------------
from src.orchestrator.artifacts.store import ArtifactStore
from src.orchestrator.config.schemas import IOConfig, RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.core.registry import PipelineRegistry
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.orchestrator.logging.setup import build_run_logger
from src.orchestrator.runtime import discovery

# ---------------- Marketdata imports ----------------
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market

# ---------------- Portfolio / pricing imports ----------------
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer

# ---------------- Instrument imports ----------------
from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.instruments.fx.options.digital import EuropeanFxDigitalOption

# ---------------- Plotting imports ----------------
from src.core.reporting.plots.portfolio.portfolio import (
    PlotOptions,
    SpotSweepResult,
    plot_positions_pv_vs_spot,
    plot_position_pv_vs_spot,
    plot_total_greek_vs_spot,
    plot_total_pnl_vs_spot,
    plot_total_pv_vs_spot,
)


# =============================================================================
# Small utilities (keep the example tidy + robust)
# =============================================================================

def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime (consistent timestamping)."""
    return datetime.now(timezone.utc)


def print_header(title: str) -> None:
    """Print a readable console header (only used at end of script)."""
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def _safe_get(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Safe dict get without noisy KeyError handling at call sites."""
    return d[key] if key in d else default


def _pretty_kv(title: str, items: Iterable[Tuple[str, Any]]) -> None:
    """Pretty-print simple key/value pairs (console output only)."""
    print_header(title)
    for k, v in items:
        print(f"{k}: {v}")


# =============================================================================
# Canonical IDs + portfolio builder
# =============================================================================

@dataclass(frozen=True, slots=True)
class FxMarketIds:
    """
    Canonical string IDs used across:
      - marketdata pipeline config (strings)
      - portfolio instruments (parsed MarketId objects)
    """
    spot: str
    vol: str
    rd: str
    rf: str


def build_fx_market_id_strings() -> FxMarketIds:
    """
    Build MarketId strings in the exact format required by MarketId.parse(...).

    IMPORTANT
    ---------
    Do NOT use str(MarketId(...)) in configs. That prints a dataclass repr
    like "MarketId(asset_class='FX', ...)" which MarketId.parse() cannot read.
    """
    return FxMarketIds(
        spot="FX.SPOT.EURUSD",
        vol="FX.VOL.EURUSD.ATM",
        rd="IR.CURVE.USD.OIS|ccy=USD",
        rf="IR.CURVE.EUR.OIS|ccy=EUR",
    )


def build_fx_example_portfolio(*, market: Market, ids: FxMarketIds) -> Portfolio:
    """
    Build a small FX portfolio using Position objects.

    Key idea
    --------
    - Position owns identity (position_id) and quantity.
    - Instruments remain pure product definitions (no trade_id field required).
    """
    # Parse string ids once into MarketId objects (instruments expect MarketId, not strings).
    spot_id = MarketId.parse(ids.spot)
    vol_id = MarketId.parse(ids.vol)
    rd_id = MarketId.parse(ids.rd)
    rf_id = MarketId.parse(ids.rf)

    # Pull spot from the snapshot to set sensible strikes (ATM / OTM / ITM).
    spot = float(market.quote(spot_id))

    # Construct positions explicitly so the example is easy to read/edit.
    positions = [
        # ---------------- Linear: Spot ----------------
        Position(
            position_id="SPOT_1",
            instrument=FxSpot(
                spot_id=spot_id,
                contract_multiplier=1.0,  # explicit for clarity
            ),
            quantity=100_000.0,  # long EUR 100k vs USD (EURUSD quote)
        ),

        # ---------------- Linear: Forward ----------------
        Position(
            position_id="FWD_6M",
            instrument=FxForward(
                spot_id=spot_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
                expiry=0.5,            # years
                notional=1_000_000.0,  # contract size in your instrument schema
                strike=1.10 * spot,    # slightly "OTM" forward (just for curve shape in plots)
            ),
            quantity=1.0,  # number of forward contracts
        ),

        # ---------------- Nonlinear: Vanilla option ----------------
        Position(
            position_id="CALL_ATM_1Y",
            instrument=EuropeanFxVanillaOption(
                option_type="call",
                notional=1_000_000.0,
                strike=spot,
                expiry=1.0,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=1.0,
        ),

        Position(
            position_id="PUT_OTM_6M",
            instrument=EuropeanFxVanillaOption(
                option_type="put",
                notional=500_000.0,
                strike=0.98 * spot,
                expiry=0.5,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=2.0,
        ),

        # ---------------- Nonlinear: Digital option ----------------
        Position(
            position_id="DIGITAL_CALL_3M",
            instrument=EuropeanFxDigitalOption(
                option_type="call",
                payoff="cash",
                payout_amount=10_000.0,  # pays domestic cash if ITM
                strike=1.01 * spot,
                expiry=0.25,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=3.0,
        ),
    ]

    # Wrap into a Portfolio container.
    return Portfolio(positions=positions)


# =============================================================================
# Market override for sweeps (clean, avoids mutating Market internals)
# =============================================================================

class MarketOverride:
    """
    Lightweight wrapper around Market that overrides a single quote (spot).

    Why this exists
    ---------------
    Your Market implementation stores quotes as Quote objects (with a .value field).
    During a sweep, it is tempting to "just write a float" into market.quotes[spot_id],
    but that breaks Market.quote(...) which expects Quote objects.

    Instead, we wrap the Market and intercept quote(...) for the single spot id.
    Everything else delegates to the underlying Market.
    """

    def __init__(self, base: Market, *, spot_id: MarketId, spot_value: float) -> None:
        self._base = base                     # underlying Market snapshot (curves/vols/quotes)
        self._spot_id = spot_id               # the quote key we override (spot)
        self._spot_value = float(spot_value)  # overridden spot value for this sweep point

    @property
    def asof(self) -> str:
        # Preserve the same as-of date as the base snapshot.
        return self._base.asof

    def quote(self, mkt_id: MarketId) -> float:
        # Intercept only the spot id we are sweeping.
        if mkt_id == self._spot_id:
            return float(self._spot_value)
        # Delegate other quotes to the base market.
        return float(self._base.quote(mkt_id))

    def curve(self, mkt_id: MarketId):
        # Delegate curves unchanged.
        return self._base.curve(mkt_id)

    def vol_surface(self, mkt_id: MarketId):
        # Delegate vol surfaces unchanged.
        return self._base.vol_surface(mkt_id)

    def __getattr__(self, name: str):
        # Defensive fallback: forward any other attribute access to the base Market.
        return getattr(self._base, name)


# =============================================================================
# Spot sweep (PV / Greeks across spot grid)
# =============================================================================

def price_spot_sweep(
    *,
    portfolio: Portfolio,
    portfolio_pricer: PortfolioPricer,
    market: Market,
    spot_id: MarketId,
    spot_grid: np.ndarray,
    greek_keys: Iterable[str],
    capture_position_ids: Iterable[str],
) -> SpotSweepResult:
    """
    Price a portfolio across a grid of spot values.

    Notes
    -----
    - Uses MarketOverride to avoid mutating the underlying Market internals.
    - Assumes curves/vol surfaces stay fixed while spot varies (standard "spot bump" view).
    """
    # Normalize inputs to deterministic numpy arrays.
    spot_grid = np.asarray(spot_grid, dtype=float).reshape(-1)

    if spot_grid.size == 0:
        raise ValueError("spot_grid must not be empty.")

    # Base spot from the snapshot (used for annotations and PnL baseline).
    spot0 = float(market.quote(spot_id))

    # Pre-allocate arrays for totals (fast, predictable memory).
    pv_total = np.zeros(spot_grid.shape[0], dtype=float)

    # Pre-allocate arrays for requested greeks (each greek gets its own vector).
    greek_keys = [str(k) for k in greek_keys]
    greeks_total: Dict[str, np.ndarray] = {k: np.zeros(spot_grid.shape[0], dtype=float) for k in greek_keys}

    # Pre-allocate arrays for per-position PV curves.
    capture_position_ids = [str(pid) for pid in capture_position_ids]
    pv_by_position: Dict[str, np.ndarray] = {
        pid: np.zeros(spot_grid.shape[0], dtype=float) for pid in capture_position_ids
    }

    # Main sweep loop (intentionally explicit; easy to debug).
    for i, s in enumerate(spot_grid):
        # Wrap the market so only the spot quote changes (curves/vols remain identical).
        mkt_s = MarketOverride(market, spot_id=spot_id, spot_value=float(s))

        # Price the full portfolio at this spot.
        res = portfolio_pricer.price(portfolio, mkt_s)

        # Store total PV.
        pv_total[i] = float(res.totals.pv)

        # Store selected total greeks (missing -> 0.0 for robustness).
        totals_greeks = getattr(res.totals, "greeks", None) or {}
        for k in greek_keys:
            greeks_total[k][i] = float(totals_greeks.get(k, 0.0))

        # Store PV for selected positions.
        for r in getattr(res, "per_position", []):
            if r.position_id in pv_by_position:
                pv_by_position[r.position_id][i] = float(r.pv)

    # Package results into your reporting container (used by plot helpers).
    return SpotSweepResult(
        spot0=spot0,
        spot_grid=spot_grid,
        pv_total=pv_total,
        greeks_total=greeks_total,
        pv_by_position=pv_by_position,
    )


# =============================================================================
# Main orchestration
# =============================================================================

def main(*, save_files: bool = False) -> None:
    """
    Run:
      (1) marketdata.build_timeseries  -> produces dataset + market snapshot
      (2) pricing.price_portfolio     -> consumes market + portfolio, produces pricing_result
      (3) spot sweep + plots           -> uses the resolved registry for consistent pricing
    """
    # ----------------- Run identity / workdir -----------------
    run_id = f"fx_marketdata_and_pricing_{utc_now().strftime('%Y%m%d_%H%M%S')}"
    workdir = Path("./.runs").expanduser().resolve()

    # ----------------- Canonical IDs as STRINGS (config-safe) -----------------
    fx_ids = build_fx_market_id_strings()

    # ----------------- Choose a small time window for fast runs -----------------
    end_dt = utc_now().date()
    start_dt = (utc_now() - timedelta(days=2)).date()

    # ----------------- Shared params dict (marketdata block is required here) -----------------
    # We keep params in one place and let each pipeline read its own block.
    params: Dict[str, Any] = {
        "marketdata": {
            "provider": {
                "type": "synthetic",
                "seed": 123,
                "name": "SyntheticProvider",
            },
            "ids": [fx_ids.spot, fx_ids.vol, fx_ids.rd, fx_ids.rf],
            "start": str(start_dt),
            "end": str(end_dt),
            "freq": "D",
            "scenarios": 1,
            "snapshot": {
                "time": "last",
                "scenario_idx": 0,
            },
        },
        # Pricing block is optional in V1 (your pricing pipeline currently ignores most config),
        # but we reserve it here for future growth (named pricers, overrides, etc.).
        "pricing": {
            "registry": {"kind": "default"},
        },
    }

    # ----------------- RunConfig for marketdata pipeline -----------------
    # Note: cfg.pipeline is the registry key used to resolve which pipeline to build.
    md_cfg = RunConfig(
        pipeline="marketdata.build_timeseries",
        io=IOConfig(workdir=str(workdir)),
        params=params,
    )

    # ----------------- ArtifactStore carrier (optional disk layout) -----------------
    store = ArtifactStore(
        workdir=workdir,
        run_id=run_id,
        artifacts_dirname=str(md_cfg.io.artifacts_dir),
        logs_dirname=str(md_cfg.io.logs_dir),
    )
    if save_files:
        store.ensure_layout()

    # ----------------- Logger (console always; file only if save_files=True) -----------------
    logger = build_run_logger(
        logger_name="QuantStrata.Examples.FxMarketdataPortfolioPricing",
        log_file=(store.logs_root / "run.log") if save_files else None,
    )

    # ----------------- Pipeline registry + discovery -----------------
    # This discovers BOTH marketdata and pricing pipelines (as you already implemented).
    registry = PipelineRegistry()
    discovery.register_builtin_pipelines(registry)

    # ----------------- Build + run marketdata pipeline -----------------
    md_builder = registry.get(md_cfg.pipeline)      # resolve builder by name
    md_pipeline = md_builder(md_cfg)               # construct Pipeline object

    # Create context for marketdata execution (state starts empty).
    md_ctx = Context(
        run_id=run_id,
        cfg=md_cfg,
        logger=logger,
        artifact_store=store,
        provider=None,
        state={},
    )

    runner = PipelineRunner(only=None, skip=None, resume_from=None, dry_run=False)
    md_ctx_final = runner.run(md_pipeline, md_ctx)

    # ----------------- Extract snapshot Market (required for pricing) -----------------
    market: Optional[Market] = _safe_get(md_ctx_final.state, Keys.MARKET, None)
    if market is None:
        raise RuntimeError(
            f"Market snapshot missing from ctx.state[{Keys.MARKET!r}]. "
            "Did you set params['marketdata']['snapshot'] in config?"
        )

    # ----------------- Build portfolio (Position-based) -----------------
    # We build instruments from the snapshot so strikes are consistent with spot0.
    portfolio = build_fx_example_portfolio(market=market, ids=fx_ids)

    # -------------------------------------------------------------------------
    # Run pricing pipeline (THIS is the integration you asked for)
    # -------------------------------------------------------------------------
    # The pricing pipeline assumes:
    #   ctx.state["market"] exists  (produced by marketdata pipeline)
    #   ctx.state["portfolio"] exists (we inject it here)
    md_ctx_final.put(Keys.PORTFOLIO, portfolio)

    # Create a pricing RunConfig (separate pipeline identity; same params/workdir).
    px_cfg = RunConfig(
        pipeline="pricing.price_portfolio",
        io=md_cfg.io,           # reuse the same IO/workdir settings
        params=params,          # same params dict (pricing reads params["pricing"])
    )

    # Build the pricing pipeline from registry.
    px_builder = registry.get(px_cfg.pipeline)
    px_pipeline = px_builder(px_cfg)

    # IMPORTANT:
    # We run pricing using a Context that *shares* the same state dict so the pipeline
    # can read "market" and we can keep all outputs in one place afterwards.
    px_ctx = Context(
        run_id=run_id,
        cfg=px_cfg,
        logger=logger,                    # same logger -> consistent log stream
        artifact_store=store,             # same store -> consistent run folder
        provider=md_ctx_final.provider,   # carry provider (even if pricing doesn't use it)
        state=md_ctx_final.state,         # share state between pipelines
    )

    # Execute pricing pipeline (now you will see STEP_START/END for pricing too).
    final_ctx = runner.run(px_pipeline, px_ctx)

    # Pull pricing outputs from stable keys defined by your pricing pipeline.
    pricing_result = final_ctx.get(Keys.PORTFOLIO_PRICING_RESULT)
    pricing_summary = final_ctx.get(Keys.PORTFOLIO_PRICING_SUMMARY)
    pricer_registry = final_ctx.get(Keys.PRICER_REGISTRY)

    # -------------------------------------------------------------------------
    # Spot sweep + plots
    # -------------------------------------------------------------------------
    # For the sweep we *reuse* the exact same registry returned by the pricing pipeline,
    # so pricing is consistent between "base" and "sweep".
    portfolio_pricer = PortfolioPricer(pricer_registry=pricer_registry)

    # Spot MarketId used by instruments and Market quote lookup.
    spot_id = MarketId.parse(fx_ids.spot)

    # Base spot from snapshot (used for grid construction and printing).
    spot0 = float(market.quote(spot_id))

    # Build a sensible spot grid around spot0.
    bump_pct = 0.10   # +/-10% spot sweep
    n_points = 25     # resolution of sweep curve
    spot_grid = np.linspace(
        (1.0 - bump_pct) * spot0,
        (1.0 + bump_pct) * spot0,
        int(n_points),
        dtype=float,
    )

    # Greeks we want to plot if available from pricers.
    greek_keys = ["delta", "gamma", "vega"]

    # Positions we want to show on the per-position PV plot(s).
    position_ids_to_plot = ["CALL_ATM_1Y", "PUT_OTM_6M", "DIGITAL_CALL_3M", "FWD_6M", "SPOT_1"]

    # log sweep start/end for visibility (no prints; keeps output clean).
    if logger is not None:
        logger.info(
            "SPOT_SWEEP_START | spot0=%.8f | bump=+/-%.2f%% | n_points=%d",
            spot0,
            100.0 * bump_pct,
            n_points,
        )

    sweep = price_spot_sweep(
        portfolio=portfolio,
        portfolio_pricer=portfolio_pricer,
        market=market,
        spot_id=spot_id,
        spot_grid=spot_grid,
        greek_keys=greek_keys,
        capture_position_ids=position_ids_to_plot,
    )

    if logger is not None:
        logger.info("SPOT_SWEEP_END | computed=%d points", int(sweep.spot_grid.shape[0]))

    # Plot options (save into outputs folder if save_files=True).
    plot_cfg = PlotOptions(
        show=True,               # set False in CI
        save=bool(save_files),   # write files only when requested
        out_dir=Path("outputs/orchestrator/pricing_spot_sweep"),
        dpi=180,
        block=True,
        close=False,
    )

    # Total portfolio plots.
    plot_total_pv_vs_spot(sweep=sweep, cfg=plot_cfg)
    plot_total_pnl_vs_spot(sweep=sweep, cfg=plot_cfg)

    # Greek plots (each is a separate clean figure).
    plot_total_greek_vs_spot(sweep=sweep, greek_key="delta", cfg=plot_cfg)
    plot_total_greek_vs_spot(sweep=sweep, greek_key="gamma", cfg=plot_cfg)
    plot_total_greek_vs_spot(sweep=sweep, greek_key="vega", cfg=plot_cfg)

    # Per-position plots (use sparingly).
    plot_positions_pv_vs_spot(sweep=sweep, position_ids=position_ids_to_plot, cfg=plot_cfg)
    plot_position_pv_vs_spot(sweep=sweep, position_id="CALL_ATM_1Y", cfg=plot_cfg)
    plot_position_pv_vs_spot(sweep=sweep, position_id="FWD_6M", cfg=plot_cfg)
    plot_position_pv_vs_spot(sweep=sweep, position_id="SPOT_1", cfg=plot_cfg)

    # =============================================================================
    # IMPORTANT: all plain print output is at the end, AFTER all logging is complete
    # =============================================================================

    print_header("RUN INFO")
    print(f"run_id     : {run_id}")
    print(f"workdir    : {workdir}")
    print(f"save_files : {save_files}")
    print()
    print("PIPELINES RUN")
    print("=============")
    print(f"- {md_cfg.pipeline}")
    print(f"- {px_cfg.pipeline}")

    # State is shared, so it doesn't matter which ctx we reference here.
    state = final_ctx.state

    request_summary = _safe_get(state, Keys.REQUEST_SUMMARY, None)
    market_ids_pretty = _safe_get(state, Keys.MARKET_IDS_PRETTY, None)
    snapshot_summary = _safe_get(state, Keys.MARKET_SNAPSHOT_SUMMARY, None)

    print_header("MARKETDATA STATE KEYS")
    print(sorted(state.keys()))

    if request_summary:
        _pretty_kv(
            "REQUEST SUMMARY",
            [
                ("start", request_summary.get("start")),
                ("end", request_summary.get("end")),
                ("freq", request_summary.get("freq")),
                ("scenarios", request_summary.get("scenarios")),
                ("n_ids", request_summary.get("n_ids")),
            ],
        )

    if market_ids_pretty:
        print_header("REQUESTED MARKET IDS")
        for s in market_ids_pretty:
            print(f"- {s}")

    if snapshot_summary:
        _pretty_kv(
            "SNAPSHOT SUMMARY",
            [
                ("time_idx", snapshot_summary.get("time_idx")),
                ("scenario_idx", snapshot_summary.get("scenario_idx")),
                ("T", snapshot_summary.get("T")),
            ],
        )

    # Pricing output at base spot (from pricing pipeline output).
    print_header("PORTFOLIO PRICING (BASE SPOT)")
    print(f"asof : {pricing_summary.get('asof')}")
    print(f"spot0: {spot0:.8f}")
    print()

    print("Per-position:")
    for r in pricing_result.per_position:
        print(f"  {r.position_id:>18s} | qty={r.quantity: .6f} | pv={r.pv: .6f}")
        if getattr(r, "greeks", None):
            g = ", ".join(f"{k}={v: .6e}" for k, v in sorted(r.greeks.items()))
            print(f"    greeks: {g}")

    print()
    print(f"TOTAL PV: {pricing_result.totals.pv:,.6f}")
    if getattr(pricing_result.totals, "greeks", None):
        print("TOTAL GREEKS:")
        for k, v in sorted(pricing_result.totals.greeks.items()):
            print(f"  {k:>14s}: {v: .10f}")

    # Sweep summary (small, useful sanity check).
    print_header("SPOT SWEEP SUMMARY")
    print(f"bump_pct : +/-{bump_pct:.2%}")
    print(f"n_points : {n_points}")
    print(f"spot_grid: [{float(spot_grid[0]):.6f} .. {float(spot_grid[-1]):.6f}]")

    print_header("DONE")
    print("Marketdata + pricing (pipeline) + plots run completed successfully.")


if __name__ == "__main__":
    main(save_files=False)