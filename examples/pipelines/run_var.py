#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: risk.compute_var
===============================================================================

This script demonstrates how to use the `risk.compute_var` pipeline to compute
Value-at-Risk (VaR) using multiple methodologies.

Learning Objectives
-------------------
1. **VaR Fundamentals**: Understand VaR as a risk measure
2. **Multiple Methods**: Compare Historical, Parametric, and Monte Carlo VaR
3. **Expected Shortfall**: Understand CVaR as a coherent risk measure
4. **Pipeline Integration**: Use orchestrator for risk workflows

What This Pipeline Does
-----------------------
1. Loads portfolio and market from state
2. Loads/generates historical returns data
3. Computes Historical Simulation VaR
4. Computes Parametric (Delta-Normal) VaR
5. Computes Monte Carlo VaR
6. Computes Expected Shortfall (CVaR)
7. Compares methods and writes report

Mathematical Framework
----------------------
Value-at-Risk Definition:
    VaR_α = inf{x : P(L > x) ≤ 1 - α}
    
    "The loss that will NOT be exceeded with probability α"

For α = 95%:
    VaR_95% = 1.645 × σ (parametric, normal assumption)

Expected Shortfall (CVaR):
    ES_α = E[L | L > VaR_α]
    
    "The average loss when VaR is exceeded"

VaR Methods
-----------
1. **Historical Simulation**: Use actual historical returns
   - Pros: No distribution assumptions, captures fat tails
   - Cons: Limited by historical data, backward-looking

2. **Parametric (Delta-Normal)**: Assume normal distribution
   - Pros: Fast, closed-form, easy to decompose
   - Cons: Underestimates tail risk, ignores gamma/convexity

3. **Monte Carlo**: Simulate future scenarios
   - Pros: Flexible, can model any distribution
   - Cons: Computationally intensive, model-dependent

Production Context
------------------
At a hedge fund:
- VaR is computed daily (and intraday) for risk limits
- Regulatory VaR (Basel III) uses 99% 10-day VaR
- Internal VaR typically uses 95% or 99% 1-day
- ES is preferred as it's a coherent risk measure

Prerequisites
-------------
- Understanding of examples/fundamentals/ and examples/risk/
- Familiarity with orchestrator framework

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/pipelines/run_var.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# Orchestrator framework
# -----------------------------------------------------------------------------
from src.orchestrator.config.schemas import RunConfig, IOConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.core.state_keys import StateKeys as Keys

# -----------------------------------------------------------------------------
# Market data and instruments
# -----------------------------------------------------------------------------
from src.marketdata.core.market import Market
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.portfolio.core import Portfolio, Position
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# =============================================================================
# MARKET AND PORTFOLIO BUILDERS
# =============================================================================

def build_market() -> Market:
    """
    Build market snapshot for VaR computation.
    
    Returns
    -------
    Market
        Market snapshot with spots, curves, and vols.
    """
    return Market(
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


def build_portfolio() -> Portfolio:
    """
    Build a sample portfolio for VaR computation.
    
    Returns
    -------
    Portfolio
        Portfolio with large directional and hedging positions.
    """
    positions = [
        # Large directional position
        Position(
            position_id="EURUSD_LONG_CALL",
            instrument=EuropeanFxVanillaOption(
                option_type="call",
                notional=50_000_000,  # $50M notional
                strike=1.10,
                expiry=0.5,
                spot_id=MarketId.parse("FX.SPOT.EURUSD"),
                vol_id=MarketId.parse("FX.VOL.EURUSD"),
                domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
                foreign_curve_id=MarketId.parse("IR.ZERO.EUR"),
            ),
            quantity=1,
        ),
        
        # Partially hedging put
        Position(
            position_id="EURUSD_LONG_PUT",
            instrument=EuropeanFxVanillaOption(
                option_type="put",
                notional=30_000_000,
                strike=1.05,
                expiry=0.5,
                spot_id=MarketId.parse("FX.SPOT.EURUSD"),
                vol_id=MarketId.parse("FX.VOL.EURUSD"),
                domestic_curve_id=MarketId.parse("IR.ZERO.USD"),
                foreign_curve_id=MarketId.parse("IR.ZERO.EUR"),
            ),
            quantity=1,
        ),
    ]
    
    return Portfolio(positions=positions)


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config() -> RunConfig:
    """
    Build configuration for VaR computation.
    
    Returns
    -------
    RunConfig
        Validated configuration for the VaR pipeline.
    """
    config = RunConfig(
        pipeline="risk.compute_var",
        
        io=IOConfig(workdir="./artifacts/var_example"),
        
        params={
            "var": {
                # Confidence levels to compute
                "confidence_levels": [0.95, 0.99],
                
                # VaR horizon (in days)
                "horizon_days": 1,
                
                # Method-specific settings
                "methods": {
                    "historical": {
                        "enabled": True,
                        "lookback_days": 252,    # 1 year of history
                        "decay": 0.94,           # Exponential decay for weighting
                    },
                    "parametric": {
                        "enabled": True,
                        "covariance": "exponential",  # How to estimate covariance
                        "decay": 0.94,
                    },
                    "monte_carlo": {
                        "enabled": True,
                        "n_simulations": 10000,
                        "model": "gbm",          # GBM, GARCH, or historical_bootstrap
                    },
                },
                
                # Also compute Expected Shortfall (CVaR)
                "compute_es": True,
                
                # Decompose VaR by risk factor
                "decomposition": True,
            }
        },
    )
    
    return validate_run_config(config)


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def display_var_results(
    historical_var: Dict[float, float],
    parametric_var: Dict[float, float],
    monte_carlo_var: Dict[float, float],
) -> None:
    """Display VaR results by method."""
    logger.info("")
    logger.info("VaR by Method (1-Day Horizon):")
    logger.info("-" * 70)
    logger.info(f"{'Method':<20} {'VaR(95%)':>15} {'VaR(99%)':>15}")
    logger.info("-" * 70)
    
    # Historical VaR
    h95 = historical_var.get(0.95, 0)
    h99 = historical_var.get(0.99, 0)
    logger.info(f"{'Historical':<20} ${h95:>14,.0f} ${h99:>14,.0f}")
    
    # Parametric VaR
    p95 = parametric_var.get(0.95, 0)
    p99 = parametric_var.get(0.99, 0)
    logger.info(f"{'Parametric':<20} ${p95:>14,.0f} ${p99:>14,.0f}")
    
    # Monte Carlo VaR
    m95 = monte_carlo_var.get(0.95, 0)
    m99 = monte_carlo_var.get(0.99, 0)
    logger.info(f"{'Monte Carlo':<20} ${m95:>14,.0f} ${m99:>14,.0f}")
    
    logger.info("-" * 70)


def display_expected_shortfall(expected_shortfall: Dict[str, Dict[float, float]]) -> None:
    """Display Expected Shortfall results."""
    logger.info("")
    logger.info("Expected Shortfall (CVaR):")
    logger.info("-" * 50)
    
    es_historical = expected_shortfall.get("historical", {})
    es95 = es_historical.get(0.95, 0)
    es99 = es_historical.get(0.99, 0)
    
    logger.info(f"  ES(95%): ${es95:>12,.0f}")
    logger.info(f"  ES(99%): ${es99:>12,.0f}")


def display_method_comparison(
    historical_var: Dict[float, float],
    parametric_var: Dict[float, float],
    monte_carlo_var: Dict[float, float],
) -> None:
    """Display method comparison."""
    h95 = historical_var.get(0.95, 0)
    p95 = parametric_var.get(0.95, 0)
    m95 = monte_carlo_var.get(0.95, 0)
    
    h99 = historical_var.get(0.99, 0)
    p99 = parametric_var.get(0.99, 0)
    m99 = monte_carlo_var.get(0.99, 0)
    
    logger.info("")
    logger.info("Method Comparison:")
    logger.info("-" * 70)
    
    var_95_values = [h95, p95, m95]
    
    if any(var_95_values):
        most_conservative = max(var_95_values)
        least_conservative = min(var_95_values)
        spread = most_conservative - least_conservative
        
        logger.info(f"  VaR(95%) spread:    ${spread:,.0f} between methods")
        logger.info(f"  Most conservative:  ${most_conservative:,.0f}")
        logger.info(f"  Least conservative: ${least_conservative:,.0f}")


def display_interpretation(
    historical_var: Dict[float, float],
    expected_shortfall: Dict[str, Dict[float, float]],
    portfolio_value: float,
) -> None:
    """Display VaR interpretation."""
    h95 = historical_var.get(0.95, 0)
    h99 = historical_var.get(0.99, 0)
    
    es_historical = expected_shortfall.get("historical", {})
    es95 = es_historical.get(0.95, 0)
    
    logger.info("")
    logger.info("Interpretation:")
    logger.info("-" * 70)
    logger.info(f"  At 95% confidence, daily losses should not exceed ~${h95:,.0f}")
    logger.info(f"  At 99% confidence, daily losses should not exceed ~${h99:,.0f}")
    logger.info("")
    logger.info(f"  However, when VaR is exceeded (5% of days at 95% level),")
    logger.info(f"  the AVERAGE loss (ES) is ~${es95:,.0f}")
    
    if portfolio_value > 0:
        logger.info("")
        logger.info(f"  VaR(95%) as % of portfolio: {h95 / portfolio_value * 100:.2f}%")
        logger.info(f"  VaR(99%) as % of portfolio: {h99 / portfolio_value * 100:.2f}%")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary() -> None:
    """Print summary of key concepts."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    summary = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         KEY TAKEAWAYS                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. VaR Definition:                                                 │
    │     - "Maximum loss at a given confidence level"                    │
    │     - VaR(95%) = loss NOT exceeded 95% of the time                  │
    │                                                                      │
    │  2. VaR Methods:                                                    │
    │     - Historical: Uses actual returns, no distribution assumption   │
    │     - Parametric: Assumes normal, fast but underestimates tails     │
    │     - Monte Carlo: Flexible but computationally intensive           │
    │                                                                      │
    │  3. Expected Shortfall:                                             │
    │     - ES = average loss WHEN VaR is exceeded                        │
    │     - More conservative and coherent than VaR                       │
    │                                                                      │
    │  4. Production Use:                                                 │
    │     - Daily risk limits and breach monitoring                       │
    │     - Regulatory capital (Basel III)                                │
    │     - Method comparison for model risk                              │
    │                                                                      │
    │  NEXT: See run_build_curves.py for curve bootstrapping              │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Execute the VaR pipeline and display results.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    logger.info("=" * 70)
    logger.info("Pipeline Example: risk.compute_var")
    logger.info("=" * 70)
    
    try:
        # ---------------------------------------------------------------------
        # Step 1: Build prerequisites
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[1/5] Building market and portfolio...")
        market = build_market()
        portfolio = build_portfolio()
        
        # Calculate approximate portfolio value
        portfolio_value = sum(
            p.instrument.notional * 0.02  # Rough option premium estimate
            for p in portfolio.positions
        )
        
        logger.info(f"      Portfolio positions: {len(portfolio.positions)}")
        logger.info(f"      Approximate PV: ${portfolio_value:,.0f}")
        
        # ---------------------------------------------------------------------
        # Step 2: Build configuration
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[2/5] Building configuration...")
        cfg = build_config()
        logger.info(f"      Pipeline: {cfg.pipeline}")
        logger.info(f"      Confidence levels: 95%, 99%")
        logger.info(f"      Methods: Historical, Parametric, Monte Carlo")
        
        # ---------------------------------------------------------------------
        # Step 3: Execute the pipeline
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[3/5] Executing pipeline...")
        
        initial_state = {
            Keys.MARKET: market,
            Keys.PORTFOLIO: portfolio,
        }
        
        ctx = run_pipeline_from_config(cfg, initial_state=initial_state)
        logger.info("      Pipeline completed successfully!")
        
        # ---------------------------------------------------------------------
        # Step 4: Extract results
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[4/5] Extracting results...")
        
        historical_var = ctx.state.get(Keys.HISTORICAL_VAR, {})
        parametric_var = ctx.state.get(Keys.PARAMETRIC_VAR, {})
        monte_carlo_var = ctx.state.get(Keys.MONTE_CARLO_VAR, {})
        expected_shortfall = ctx.state.get(Keys.EXPECTED_SHORTFALL, {})
        
        # ---------------------------------------------------------------------
        # Step 5: Display VaR Report
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[5/5] Value-at-Risk Report")
        logger.info("=" * 70)
        
        display_var_results(historical_var, parametric_var, monte_carlo_var)
        display_expected_shortfall(expected_shortfall)
        display_method_comparison(historical_var, parametric_var, monte_carlo_var)
        display_interpretation(historical_var, expected_shortfall, portfolio_value)
        
        logger.info("")
        logger.info(f"Artifacts saved to: {cfg.io.workdir}")
        
        # Summary
        print_summary()
        
        logger.info("Pipeline example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VaR Computation Pipeline Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    args = parser.parse_args()
    main(args)
