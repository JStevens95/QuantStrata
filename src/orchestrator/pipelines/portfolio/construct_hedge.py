"""
Pipeline: portfolio.construct_hedge

Construct a hedge portfolio to neutralise specific Greek exposures.

Purpose
-------
Build a hedge portfolio that neutralises target Greeks (delta, vega, etc.) by:
1. Loading portfolio from state
2. Computing current Greek exposures
3. Defining target Greeks (e.g., delta=0, vega=0)
4. Selecting available hedging instruments
5. Optimising hedge quantities via least squares
6. Building hedge portfolio
7. Validating residual Greeks within tolerance

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
from src.portfolio.core import Portfolio, Position
from src.marketdata.core.ids import MarketId


# =============================================================================
# Configuration Helpers
# =============================================================================

def _require_dict(parent: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Extract a required dictionary from parent mapping."""
    if key not in parent:
        raise KeyError(f"Missing required config key: '{key}'")
    value = parent[key]
    if not isinstance(value, dict):
        raise TypeError(f"Config key '{key}' must be a dict")
    return value


def _hedge_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'hedge' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return _require_dict(cfg.params, "hedge")


# =============================================================================
# Pipeline Steps
# =============================================================================

@dataclass(slots=True)
class LoadPortfolioStep(Step):
    """Step 1: Load portfolio to hedge from state."""
    
    def run(self, ctx: Context) -> Context:
        # Verify portfolio exists in state
        if Keys.PORTFOLIO not in ctx.state:
            raise KeyError("Missing ctx.state['portfolio']. Run portfolio.build_from_config first.")
        
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        
        if ctx.logger:
            ctx.logger.info("Loaded portfolio with %d positions for hedging", len(portfolio))
        
        return ctx


@dataclass(slots=True)
class ComputeGreeksStep(Step):
    """Step 2: Compute current Greek exposures."""
    
    def run(self, ctx: Context) -> Context:
        # Get portfolio and market
        portfolio: Portfolio = ctx.get(Keys.PORTFOLIO)
        
        if Keys.MARKET not in ctx.state:
            raise KeyError("Missing ctx.state['market']. Need market snapshot for Greeks.")
        
        market = ctx.get(Keys.MARKET)
        
        # Compute aggregate Greeks (simplified - use real Greeks computation in production)
        portfolio_greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }
        
        # In production, iterate positions and compute Greeks via pricer
        # Here we use placeholder values for demonstration
        for pos in portfolio:
            # Simplified: assume each option has delta ~0.5 * quantity
            portfolio_greeks["delta"] += 0.5 * pos.quantity * 1_000_000  # Notional-adjusted
            portfolio_greeks["gamma"] += 0.02 * abs(pos.quantity) * 1_000_000
            portfolio_greeks["vega"] += 0.15 * pos.quantity * 1_000_000
            portfolio_greeks["theta"] += -0.01 * pos.quantity * 1_000_000
            portfolio_greeks["rho"] += 0.05 * pos.quantity * 1_000_000
        
        ctx.put(Keys.PORTFOLIO_GREEKS, portfolio_greeks)
        
        if ctx.logger:
            ctx.logger.info(
                "Portfolio Greeks: delta=%.2f, gamma=%.2f, vega=%.2f",
                portfolio_greeks["delta"], portfolio_greeks["gamma"], portfolio_greeks["vega"]
            )
        
        return ctx


@dataclass(slots=True)
class DefineTargetGreeksStep(Step):
    """Step 3: Define target Greeks from config."""
    
    def run(self, ctx: Context) -> Context:
        # Get target Greeks from config
        hedge_cfg = _hedge_cfg(ctx.cfg)
        target_cfg = hedge_cfg.get("target_greeks", {})
        
        # Build target Greeks dict (None = don't hedge that Greek)
        target_greeks = {
            "delta": target_cfg.get("delta"),
            "gamma": target_cfg.get("gamma"),
            "vega": target_cfg.get("vega"),
            "theta": target_cfg.get("theta"),
            "rho": target_cfg.get("rho"),
        }
        
        ctx.put(Keys.TARGET_GREEKS, target_greeks)
        
        # Log which Greeks we're hedging
        hedging = [k for k, v in target_greeks.items() if v is not None]
        if ctx.logger:
            ctx.logger.info("Target Greeks to hedge: %s", ", ".join(hedging) or "none")
        
        return ctx


@dataclass(slots=True)
class SelectHedgeInstrumentsStep(Step):
    """Step 4: Select available hedging instruments."""
    
    def run(self, ctx: Context) -> Context:
        # Get hedge instruments from config
        hedge_cfg = _hedge_cfg(ctx.cfg)
        instruments_cfg = hedge_cfg.get("hedge_instruments", [])
        
        # Build list of available hedge instruments
        # In production, would instantiate actual instrument objects
        hedge_instruments = []
        
        for i, inst_cfg in enumerate(instruments_cfg):
            inst_type = str(inst_cfg.get("type", "")).lower()
            underlying = inst_cfg.get("underlying", "EURUSD")
            
            hedge_instruments.append({
                "index": i,
                "type": inst_type,
                "underlying": underlying,
                "config": inst_cfg,
                # Simplified Greeks for hedge instruments
                "delta": 1.0 if inst_type == "fxspot" else 0.5,
                "gamma": 0.0 if inst_type == "fxspot" else 0.02,
                "vega": 0.0 if inst_type == "fxspot" else 0.15,
            })
        
        ctx.put(Keys.HEDGE_INSTRUMENTS, hedge_instruments)
        
        if ctx.logger:
            ctx.logger.info("Selected %d hedge instruments", len(hedge_instruments))
        
        return ctx


@dataclass(slots=True)
class OptimiseHedgeStep(Step):
    """Step 5: Solve for optimal hedge quantities."""
    
    def run(self, ctx: Context) -> Context:
        # Get required data
        portfolio_greeks: Dict[str, float] = ctx.get(Keys.PORTFOLIO_GREEKS)
        target_greeks: Dict[str, Optional[float]] = ctx.get(Keys.TARGET_GREEKS)
        hedge_instruments: List[Dict] = ctx.get(Keys.HEDGE_INSTRUMENTS)
        
        # Build optimisation problem
        # For each Greek to hedge: sum(quantity[i] * greek[i]) = target - current
        greeks_to_hedge = [k for k, v in target_greeks.items() if v is not None]
        
        if not greeks_to_hedge:
            ctx.put(Keys.HEDGE_QUANTITIES, {})
            return ctx
        
        if not hedge_instruments:
            raise ValueError("No hedge instruments available")
        
        # Simple least squares solution
        # A @ x = b where A[i,j] = greek_j for instrument_i, x = quantities
        n_greeks = len(greeks_to_hedge)
        n_instruments = len(hedge_instruments)
        
        A = np.zeros((n_greeks, n_instruments))
        b = np.zeros(n_greeks)
        
        for i, greek in enumerate(greeks_to_hedge):
            target = target_greeks[greek] if target_greeks[greek] is not None else 0.0
            current = portfolio_greeks.get(greek, 0.0)
            b[i] = target - current
            
            for j, inst in enumerate(hedge_instruments):
                A[i, j] = inst.get(greek, 0.0)
        
        # Solve least squares: min ||Ax - b||^2
        try:
            quantities, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            quantities = np.zeros(n_instruments)
        
        # Build quantities dict
        hedge_quantities = {
            f"hedge_{i}": float(quantities[i]) 
            for i in range(n_instruments)
        }
        
        ctx.put(Keys.HEDGE_QUANTITIES, hedge_quantities)
        
        if ctx.logger:
            ctx.logger.info(
                "Optimised hedge quantities: %s",
                ", ".join(f"{k}={v:.2f}" for k, v in hedge_quantities.items())
            )
        
        return ctx


@dataclass(slots=True)
class BuildHedgePortfolioStep(Step):
    """Step 6: Construct hedge portfolio."""
    
    def run(self, ctx: Context) -> Context:
        # Get quantities and instruments
        hedge_quantities: Dict[str, float] = ctx.get(Keys.HEDGE_QUANTITIES)
        hedge_instruments: List[Dict] = ctx.get(Keys.HEDGE_INSTRUMENTS)
        
        # Build hedge positions (simplified - would use real instruments)
        hedge_positions: List[Position] = []
        
        for i, inst in enumerate(hedge_instruments):
            qty = hedge_quantities.get(f"hedge_{i}", 0.0)
            
            if abs(qty) < 1e-6:
                continue  # Skip zero quantities
            
            # Create placeholder position
            # In production, instantiate actual instrument
            position = Position(
                position_id=f"hedge_pos_{i}",
                instrument={"type": inst["type"], "underlying": inst["underlying"]},
                quantity=qty,
            )
            hedge_positions.append(position)
        
        # Create hedge portfolio
        hedge_portfolio = Portfolio(positions=hedge_positions) if hedge_positions else None
        
        ctx.put(Keys.HEDGE_PORTFOLIO, hedge_portfolio)
        
        if ctx.logger:
            ctx.logger.info(
                "Built hedge portfolio with %d positions",
                len(hedge_positions)
            )
        
        return ctx


@dataclass(slots=True)
class ValidateHedgeStep(Step):
    """Step 7: Verify residual Greeks within tolerance."""
    
    def run(self, ctx: Context) -> Context:
        # Get data
        portfolio_greeks: Dict[str, float] = ctx.get(Keys.PORTFOLIO_GREEKS)
        target_greeks: Dict[str, Optional[float]] = ctx.get(Keys.TARGET_GREEKS)
        hedge_quantities: Dict[str, float] = ctx.get(Keys.HEDGE_QUANTITIES)
        hedge_instruments: List[Dict] = ctx.get(Keys.HEDGE_INSTRUMENTS)
        hedge_cfg = _hedge_cfg(ctx.cfg)
        
        # Get tolerances
        tolerance_cfg = hedge_cfg.get("tolerance", {})
        
        # Calculate residual Greeks
        residual_greeks: Dict[str, float] = {}
        
        for greek, target in target_greeks.items():
            if target is None:
                continue
            
            current = portfolio_greeks.get(greek, 0.0)
            
            # Add hedge contribution
            hedge_contribution = 0.0
            for i, inst in enumerate(hedge_instruments):
                qty = hedge_quantities.get(f"hedge_{i}", 0.0)
                hedge_contribution += qty * inst.get(greek, 0.0)
            
            residual = current + hedge_contribution - target
            residual_greeks[greek] = residual
            
            # Check tolerance
            tol = tolerance_cfg.get(greek, 0.01)
            if abs(residual) > tol:
                if ctx.logger:
                    ctx.logger.warning(
                        "Residual %s=%.4f exceeds tolerance %.4f",
                        greek, residual, tol
                    )
        
        ctx.put(Keys.RESIDUAL_GREEKS, residual_greeks)
        
        if ctx.logger:
            ctx.logger.info(
                "Hedge validation complete. Residuals: %s",
                ", ".join(f"{k}={v:.4f}" for k, v in residual_greeks.items())
            )
        
        return ctx


# =============================================================================
# Pipeline Builder
# =============================================================================

def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the portfolio.construct_hedge pipeline."""
    steps: List[Step] = [
        LoadPortfolioStep(name="load_portfolio"),
        ComputeGreeksStep(name="compute_greeks"),
        DefineTargetGreeksStep(name="define_target_greeks"),
        SelectHedgeInstrumentsStep(name="select_hedge_instruments"),
        OptimiseHedgeStep(name="optimise_hedge"),
        BuildHedgePortfolioStep(name="build_hedge_portfolio"),
        ValidateHedgeStep(name="validate_hedge"),
    ]
    
    return Pipeline(name="portfolio.construct_hedge", steps=steps)
