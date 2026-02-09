"""
Pipeline: portfolio.build_from_config

Construct a portfolio from a YAML/JSON position specification.

Purpose
-------
Build a Portfolio object from configuration by:
1. Parsing position specifications from config
2. Instantiating instrument objects (options, forwards, etc.)
3. Validating instrument parameters
4. Creating Position objects with quantities
5. Assembling into Portfolio object

Design Philosophy
-----------------
- Declarative: Portfolio defined entirely in config.
- Extensible: Support multiple instrument types via factory pattern.
- Validated: All instruments validated before assembly.

Author: QuantStrata Team
"""
from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

# =============================================================================
# Orchestrator Framework Imports
# =============================================================================
from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys

# =============================================================================
# Portfolio Imports
# =============================================================================
from src.portfolio.core import Portfolio, Position

# =============================================================================
# Instrument Imports
# =============================================================================
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
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


def _require_str(value: Any, *, key_name: str) -> str:
    """Validate and return a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"'{key_name}' must be a str")
    out = value.strip()
    if not out:
        raise ValueError(f"'{key_name}' must be non-empty")
    return out


def _require_float(value: Any, *, key_name: str) -> float:
    """Validate and return a float."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{key_name}' must be float-like") from exc


def _portfolio_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'portfolio' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return _require_dict(cfg.params, "portfolio")


# =============================================================================
# Instrument Factory
# =============================================================================

def _create_instrument(inst_cfg: Dict[str, Any], idx: int) -> Any:
    """
    Create an instrument from configuration.
    
    Parameters
    ----------
    inst_cfg : Dict
        Instrument configuration dictionary.
    idx : int
        Position index for error messages.
        
    Returns
    -------
    Instrument object (FxVanillaEuropeanOption, etc.)
    """
    # Get instrument type
    inst_type = _require_str(inst_cfg.get("type"), key_name=f"position[{idx}].instrument.type")
    inst_type_lower = inst_type.lower()
    
    # Route to appropriate factory
    if inst_type_lower in ("fxvanillaoption", "fx_vanilla_option", "fxvanillaeuropeanoption"):
        return _create_fx_vanilla_option(inst_cfg, idx)
    
    # Add more instrument types as needed
    raise ValueError(f"Unsupported instrument type: '{inst_type}'")


def _create_fx_vanilla_option(cfg: Dict[str, Any], idx: int) -> FxVanillaEuropeanOption:
    """
    Create an FX vanilla European option from configuration.
    """
    # Extract required fields
    underlying = _require_str(cfg.get("underlying"), key_name=f"[{idx}].underlying")
    strike = _require_float(cfg.get("strike"), key_name=f"[{idx}].strike")
    option_type = _require_str(cfg.get("option_type"), key_name=f"[{idx}].option_type").lower()
    notional = _require_float(cfg.get("notional"), key_name=f"[{idx}].notional")
    
    # Parse expiry (could be date string or year fraction)
    expiry_raw = cfg.get("expiry")
    if isinstance(expiry_raw, (int, float)):
        expiry = float(expiry_raw)
    elif isinstance(expiry_raw, str):
        # Convert date string to year fraction (simplified)
        # In production, use proper date parsing
        expiry = _require_float(cfg.get("expiry_years", 1.0), key_name=f"[{idx}].expiry_years")
    else:
        expiry = 1.0  # Default
    
    # Build market IDs (using underlying as base)
    spot_id = MarketId.parse(f"FX.SPOT.{underlying}")
    vol_id = MarketId.parse(f"FX.VOL.{underlying}")
    
    # Get curve IDs (from config or derive from underlying)
    ccy_pair = underlying.upper()
    dom_ccy = ccy_pair[3:6] if len(ccy_pair) >= 6 else "USD"
    for_ccy = ccy_pair[0:3] if len(ccy_pair) >= 3 else "EUR"
    
    dom_curve_id = MarketId.parse(cfg.get("domestic_curve", f"IR.ZERO.{dom_ccy}"))
    for_curve_id = MarketId.parse(cfg.get("foreign_curve", f"IR.ZERO.{for_ccy}"))
    
    return FxVanillaEuropeanOption(
        option_type=option_type,
        notional=notional,
        strike=strike,
        expiry=expiry,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=dom_curve_id,
        foreign_curve_id=for_curve_id,
    )


# =============================================================================
# Pipeline Steps
# =============================================================================

@dataclass(slots=True)
class ParsePositionConfigStep(Step):
    """
    Step 1: Parse position specifications from config.
    
    Extracts the positions list from portfolio configuration.
    
    Outputs
    -------
    ctx.state[Keys.POSITION_CONFIGS] : List[Dict]
        Raw position configuration dictionaries.
    """
    
    def run(self, ctx: Context) -> Context:
        # Get portfolio configuration
        portfolio_cfg = _portfolio_cfg(ctx.cfg)
        
        # Get positions list
        positions_cfg = portfolio_cfg.get("positions", [])
        if not isinstance(positions_cfg, list):
            raise TypeError("portfolio.positions must be a list")
        
        if not positions_cfg:
            raise ValueError("No positions specified in portfolio configuration")
        
        # Store raw configs
        ctx.put(Keys.POSITION_CONFIGS, positions_cfg)
        
        if ctx.logger:
            ctx.logger.info("Parsed %d position specifications", len(positions_cfg))
        
        return ctx


@dataclass(slots=True)
class BuildInstrumentsStep(Step):
    """
    Step 2: Instantiate instrument objects.
    
    Creates instrument objects from position configurations using factory.
    
    Outputs
    -------
    ctx.state[Keys.INSTRUMENTS] : List[Any]
        Instantiated instrument objects.
    """
    
    def run(self, ctx: Context) -> Context:
        # Get position configs
        position_configs: List[Dict] = ctx.get(Keys.POSITION_CONFIGS)
        
        # Build instruments
        instruments: List[Any] = []
        
        for i, pos_cfg in enumerate(position_configs):
            # Get instrument config
            inst_cfg = pos_cfg.get("instrument", {})
            if not isinstance(inst_cfg, dict):
                raise TypeError(f"position[{i}].instrument must be a dict")
            
            # Create instrument
            instrument = _create_instrument(inst_cfg, i)
            instruments.append(instrument)
        
        # Store instruments
        ctx.put(Keys.INSTRUMENTS, instruments)
        
        if ctx.logger:
            # Count by type
            type_counts: Dict[str, int] = {}
            for inst in instruments:
                t = type(inst).__name__
                type_counts[t] = type_counts.get(t, 0) + 1
            
            ctx.logger.info(
                "Built %d instruments: %s",
                len(instruments),
                ", ".join(f"{k}={v}" for k, v in type_counts.items())
            )
        
        return ctx


@dataclass(slots=True)
class ValidateInstrumentsStep(Step):
    """
    Step 3: Validate instrument parameters.
    
    Performs sanity checks on instrument parameters.
    
    Outputs
    -------
    No new state; raises on validation failure.
    """
    
    def run(self, ctx: Context) -> Context:
        # Get instruments
        instruments: List[Any] = ctx.get(Keys.INSTRUMENTS)
        
        # Validate each instrument
        for i, inst in enumerate(instruments):
            # Check for common issues
            if hasattr(inst, "strike") and inst.strike <= 0:
                raise ValueError(f"Instrument {i}: strike must be positive")
            
            if hasattr(inst, "expiry") and inst.expiry < 0:
                raise ValueError(f"Instrument {i}: expiry must be non-negative")
            
            if hasattr(inst, "notional") and inst.notional == 0:
                raise ValueError(f"Instrument {i}: notional must be non-zero")
        
        if ctx.logger:
            ctx.logger.info("Validated %d instruments: all parameters OK", len(instruments))
        
        return ctx


@dataclass(slots=True)
class BuildPositionsStep(Step):
    """
    Step 4: Create Position objects with quantities.
    
    Wraps instruments in Position objects with quantities and directions.
    
    Outputs
    -------
    ctx.state[Keys.POSITIONS] : List[Position]
        Position objects ready for portfolio assembly.
    """
    
    def run(self, ctx: Context) -> Context:
        # Get configs and instruments
        position_configs: List[Dict] = ctx.get(Keys.POSITION_CONFIGS)
        instruments: List[Any] = ctx.get(Keys.INSTRUMENTS)
        
        # Build positions
        positions: List[Position] = []
        
        for i, (pos_cfg, instrument) in enumerate(zip(position_configs, instruments)):
            # Get position ID
            position_id = _require_str(pos_cfg.get("id"), key_name=f"position[{i}].id")
            
            # Get quantity (default 1)
            quantity = float(pos_cfg.get("quantity", 1))
            
            # Get direction and adjust quantity
            direction = str(pos_cfg.get("direction", "long")).lower()
            if direction == "short":
                quantity = -abs(quantity)
            elif direction == "long":
                quantity = abs(quantity)
            
            # Create position
            position = Position(
                position_id=position_id,
                instrument=instrument,
                quantity=quantity,
            )
            positions.append(position)
        
        # Store positions
        ctx.put(Keys.POSITIONS, positions)
        
        if ctx.logger:
            long_count = sum(1 for p in positions if p.quantity > 0)
            short_count = sum(1 for p in positions if p.quantity < 0)
            ctx.logger.info(
                "Built %d positions (%d long, %d short)",
                len(positions), long_count, short_count
            )
        
        return ctx


@dataclass(slots=True)
class AssemblePortfolioStep(Step):
    """
    Step 5: Assemble positions into Portfolio object.
    
    Creates the final Portfolio object from positions.
    
    Outputs
    -------
    ctx.state[Keys.PORTFOLIO] : Portfolio
        Assembled portfolio ready for pricing.
    """
    
    def run(self, ctx: Context) -> Context:
        # Get positions
        positions: List[Position] = ctx.get(Keys.POSITIONS)
        portfolio_cfg = _portfolio_cfg(ctx.cfg)
        
        # Create portfolio
        portfolio = Portfolio(positions=positions)
        
        # Store portfolio
        ctx.put(Keys.PORTFOLIO, portfolio)
        
        # Get portfolio name for logging
        portfolio_name = portfolio_cfg.get("name", "Unnamed")
        
        if ctx.logger:
            ctx.logger.info(
                "Assembled portfolio '%s' with %d positions",
                portfolio_name, len(portfolio)
            )
        
        return ctx


# =============================================================================
# Pipeline Builder
# =============================================================================

def build_pipeline(cfg: RunConfig) -> Pipeline:
    """
    Build the portfolio.build_from_config pipeline.
    
    Parameters
    ----------
    cfg : RunConfig
        The validated run configuration.
        
    Returns
    -------
    Pipeline
        The assembled pipeline.
    """
    steps: List[Step] = [
        ParsePositionConfigStep(name="parse_position_config"),   # Step 1
        BuildInstrumentsStep(name="build_instruments"),          # Step 2
        ValidateInstrumentsStep(name="validate_instruments"),    # Step 3
        BuildPositionsStep(name="build_positions"),              # Step 4
        AssemblePortfolioStep(name="assemble_portfolio"),        # Step 5
    ]
    
    return Pipeline(
        name="portfolio.build_from_config",
        steps=steps,
    )
