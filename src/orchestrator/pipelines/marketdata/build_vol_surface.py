"""
Pipeline: marketdata.build_vol_surface

Build and validate a volatility surface from option quotes.

Purpose
-------
Construct an implied volatility surface from market vol quotes by:
1. Loading vol quotes (delta or strike convention)
2. Converting quotes to standard strike/expiry format
3. Building raw vol surface from quotes
4. Validating arbitrage constraints (calendar, butterfly)
5. Applying surface interpolation (SABR, SVI, etc.)
6. Storing the resulting VolSurface for pricing

Design Philosophy
-----------------
- Support multiple quote conventions (delta, strike, moneyness).
- Arbitrage-free validation to ensure pricing consistency.
- Configurable interpolation for different use cases.

Author: QuantStrata Team
"""
from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================
from dataclasses import dataclass  # Immutable step definitions
from typing import Any, Dict, List, Mapping, Optional, Tuple  # Type annotations

import numpy as np  # Numerical operations

# =============================================================================
# Orchestrator Framework Imports
# =============================================================================
from src.orchestrator.config.schemas import RunConfig  # Run configuration
from src.orchestrator.core.context import Context  # Execution context
from src.orchestrator.core.pipeline import Pipeline  # Pipeline container
from src.orchestrator.core.step import Step  # Step interface
from src.orchestrator.core.state_keys import StateKeys as Keys  # State keys

# =============================================================================
# Market Data Imports
# =============================================================================
from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface  # Vol surfaces


# =============================================================================
# Configuration Helpers
# =============================================================================

def _require_dict(parent: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Extract a required dictionary from parent mapping."""
    if key not in parent:
        raise KeyError(f"Missing required config key: '{key}'")
    value = parent[key]
    if not isinstance(value, dict):
        raise TypeError(f"Config key '{key}' must be a dict, got: {type(value).__name__}")
    return value


def _require_str(value: Any, *, key_name: str) -> str:
    """Validate and return a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"'{key_name}' must be a str, got: {type(value).__name__}")
    out = value.strip()
    if not out:
        raise ValueError(f"'{key_name}' must be a non-empty string")
    return out


def _require_float(value: Any, *, key_name: str) -> float:
    """Validate and return a float."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{key_name}' must be float-like") from exc


def _vol_surface_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """
    Extract the 'vol_surface' configuration block from RunConfig.params.
    """
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return _require_dict(cfg.params, "vol_surface")


# =============================================================================
# Vol Quote Data Class
# =============================================================================

@dataclass(frozen=True, slots=True)
class VolQuote:
    """
    A single volatility quote for surface construction.
    
    Attributes
    ----------
    expiry : str
        Expiry tenor string (e.g., "1M", "3M", "1Y").
    expiry_years : float
        Expiry in year fraction.
    delta : Optional[float]
        Delta value if quote is in delta convention (e.g., 0.25, 0.50).
    strike : Optional[float]
        Absolute strike if quote is in strike convention.
    vol : float
        Implied volatility (decimal, e.g., 0.10 for 10%).
    """
    expiry: str
    expiry_years: float
    delta: Optional[float]
    strike: Optional[float]
    vol: float


def _tenor_to_years(tenor: str) -> float:
    """Convert tenor string to year fraction."""
    t = tenor.upper().strip()
    
    if t.endswith("D"):
        return int(t[:-1]) / 365.0
    if t.endswith("W"):
        return int(t[:-1]) * 7 / 365.0
    if t.endswith("M"):
        return int(t[:-1]) / 12.0
    if t.endswith("Y"):
        return float(t[:-1])
    
    raise ValueError(f"Unknown tenor format: '{tenor}'")


def _parse_vol_quotes(
    quotes_config: List[Dict[str, Any]], 
    convention: str
) -> List[VolQuote]:
    """
    Parse vol quotes from configuration.
    
    Parameters
    ----------
    quotes_config : List[Dict]
        List of quote dictionaries from config.
    convention : str
        Quote convention: "delta", "strike", or "moneyness".
        
    Returns
    -------
    List[VolQuote]
        Parsed vol quotes.
    """
    quotes: List[VolQuote] = []
    
    for i, q in enumerate(quotes_config):
        # Parse expiry
        expiry = _require_str(q.get("expiry"), key_name=f"quote[{i}].expiry")
        expiry_years = _tenor_to_years(expiry)
        
        # Parse vol
        vol = _require_float(q.get("vol"), key_name=f"quote[{i}].vol")
        
        # Parse delta or strike based on convention
        delta: Optional[float] = None
        strike: Optional[float] = None
        
        if convention == "delta":
            delta = _require_float(q.get("delta"), key_name=f"quote[{i}].delta")
        elif convention == "strike":
            strike = _require_float(q.get("strike"), key_name=f"quote[{i}].strike")
        elif convention == "moneyness":
            # Moneyness will be converted to strike later
            strike = _require_float(q.get("moneyness"), key_name=f"quote[{i}].moneyness")
        
        quotes.append(VolQuote(
            expiry=expiry,
            expiry_years=expiry_years,
            delta=delta,
            strike=strike,
            vol=vol,
        ))
    
    return quotes


# =============================================================================
# Pipeline Steps
# =============================================================================

@dataclass(slots=True)
class LoadVolQuotesStep(Step):
    """
    Step 1: Load vol quotes from configuration.
    
    Reads the quotes section and parses into VolQuote objects.
    
    Outputs
    -------
    ctx.state[Keys.VOL_QUOTES] : List[VolQuote]
        Parsed vol quotes.
    """
    
    def run(self, ctx: Context) -> Context:
        # Extract vol surface configuration
        vol_cfg = _vol_surface_cfg(ctx.cfg)
        
        # Get quote convention (delta, strike, moneyness)
        convention = _require_str(
            vol_cfg.get("quote_convention", "delta"), 
            key_name="quote_convention"
        ).lower()
        
        # Get quotes list
        quotes_config = vol_cfg.get("quotes", [])
        if not isinstance(quotes_config, list):
            raise TypeError("vol_surface.quotes must be a list")
        
        if not quotes_config:
            raise ValueError("No vol quotes provided. Need at least one quote.")
        
        # Parse quotes
        quotes = _parse_vol_quotes(quotes_config, convention)
        
        # Store in state
        ctx.put(Keys.VOL_QUOTES, quotes)
        
        # Log summary
        if ctx.logger:
            expiries = set(q.expiry for q in quotes)
            ctx.logger.info(
                "Loaded %d vol quotes across %d expiries (convention: %s)",
                len(quotes), len(expiries), convention
            )
        
        return ctx


@dataclass(slots=True)
class ConvertQuotesStep(Step):
    """
    Step 2: Convert quotes to standard strike/expiry format.
    
    If quotes are in delta convention, converts to absolute strikes
    using Black-Scholes delta formula.
    
    Outputs
    -------
    Updates ctx.state[Keys.VOL_QUOTES] with converted quotes (strikes populated).
    """
    
    def run(self, ctx: Context) -> Context:
        # Get config
        vol_cfg = _vol_surface_cfg(ctx.cfg)
        convention = vol_cfg.get("quote_convention", "delta").lower()
        
        # If already in strike convention, nothing to do
        if convention == "strike":
            if ctx.logger:
                ctx.logger.info("Quotes already in strike convention; no conversion needed")
            return ctx
        
        # Get quotes and spot for delta conversion
        quotes: List[VolQuote] = ctx.get(Keys.VOL_QUOTES)
        spot = _require_float(vol_cfg.get("spot"), key_name="spot")
        
        # Get rates (optional, default to 0)
        r_dom = float(vol_cfg.get("r_domestic", 0.0))
        r_for = float(vol_cfg.get("r_foreign", 0.0))
        
        # Convert delta quotes to strikes
        converted_quotes: List[VolQuote] = []
        
        for q in quotes:
            if convention == "delta" and q.delta is not None:
                # Convert delta to strike using approximate formula
                strike = self._delta_to_strike(
                    delta=q.delta,
                    spot=spot,
                    vol=q.vol,
                    T=q.expiry_years,
                    r_dom=r_dom,
                    r_for=r_for,
                )
            elif convention == "moneyness" and q.strike is not None:
                # Moneyness to strike: K = moneyness * spot
                strike = q.strike * spot
            else:
                strike = q.strike
            
            converted_quotes.append(VolQuote(
                expiry=q.expiry,
                expiry_years=q.expiry_years,
                delta=q.delta,
                strike=strike,
                vol=q.vol,
            ))
        
        # Update state
        ctx.put(Keys.VOL_QUOTES, converted_quotes)
        
        if ctx.logger:
            ctx.logger.info("Converted %d quotes to strike format", len(converted_quotes))
        
        return ctx
    
    def _delta_to_strike(
        self, 
        delta: float, 
        spot: float, 
        vol: float, 
        T: float, 
        r_dom: float, 
        r_for: float,
    ) -> float:
        """
        Convert delta to strike using Black-Scholes formula inversion.
        
        For a call: Δ = exp(-r_for * T) * N(d1)
        For puts (delta < 0): Δ = -exp(-r_for * T) * N(-d1)
        
        Approximate inversion: K = F * exp(-vol * sqrt(T) * N^{-1}(delta * exp(r_for * T)))
        """
        from scipy.stats import norm
        
        # Forward price
        F = spot * np.exp((r_dom - r_for) * T)
        
        # Adjust delta for discounting
        if delta > 0:  # Call
            d1_target = norm.ppf(delta * np.exp(r_for * T))
        else:  # Put
            d1_target = -norm.ppf(-delta * np.exp(r_for * T))
        
        # Solve for strike: d1 = [ln(F/K) + 0.5*vol^2*T] / (vol*sqrt(T))
        # K = F * exp(-d1 * vol * sqrt(T) + 0.5 * vol^2 * T)
        sqrt_T = np.sqrt(max(T, 1e-6))
        log_moneyness = d1_target * vol * sqrt_T - 0.5 * vol * vol * T
        strike = F * np.exp(-log_moneyness)
        
        return float(strike)


@dataclass(slots=True)
class BuildRawSurfaceStep(Step):
    """
    Step 3: Build raw vol surface from quotes.
    
    Constructs a GridVolSurface from the strike/expiry/vol grid.
    
    Outputs
    -------
    ctx.state[Keys.VOL_SURFACE] : GridVolSurface
        Raw (uninterpolated) vol surface.
    """
    
    def run(self, ctx: Context) -> Context:
        # Get converted quotes
        quotes: List[VolQuote] = ctx.get(Keys.VOL_QUOTES)
        
        # Extract unique expiries and strikes
        expiries = sorted(set(q.expiry_years for q in quotes))
        strikes = sorted(set(q.strike for q in quotes if q.strike is not None))
        
        if not expiries or not strikes:
            raise ValueError("Cannot build surface: no valid expiry/strike combinations")
        
        # Build vol grid (expiry x strike)
        vol_grid = np.full((len(expiries), len(strikes)), np.nan)
        
        for q in quotes:
            if q.strike is None:
                continue
            
            # Find grid indices
            try:
                exp_idx = expiries.index(q.expiry_years)
                strike_idx = strikes.index(q.strike)
                vol_grid[exp_idx, strike_idx] = q.vol
            except ValueError:
                continue
        
        # Create GridVolSurface
        vol_surface = GridVolSurface(
            expiries=np.array(expiries),
            strikes=np.array(strikes),
            implied_vols=vol_grid,
            extrapolation="flat",
        )
        
        # Store in state
        ctx.put(Keys.VOL_SURFACE, vol_surface)
        
        if ctx.logger:
            ctx.logger.info(
                "Built raw vol surface: %d expiries x %d strikes",
                len(expiries), len(strikes)
            )
        
        return ctx


@dataclass(slots=True)
class ValidateArbitrageStep(Step):
    """
    Step 4: Validate arbitrage constraints.
    
    Checks:
    - Calendar arbitrage: Vol shouldn't decrease with expiry (for same strike)
    - Butterfly arbitrage: Vol smile should be convex in strike
    
    Outputs
    -------
    ctx.state[Keys.ARBITRAGE_REPORT] : Dict
        Arbitrage validation results.
    """
    
    def run(self, ctx: Context) -> Context:
        # Get surface and config
        vol_surface: GridVolSurface = ctx.get(Keys.VOL_SURFACE)
        vol_cfg = _vol_surface_cfg(ctx.cfg)
        
        # Check if validation is enabled
        check_arbitrage = vol_cfg.get("arbitrage_check", True)
        tolerance = float(vol_cfg.get("arbitrage_tolerance", 0.001))
        
        if not check_arbitrage:
            ctx.put(Keys.ARBITRAGE_REPORT, {"skipped": True})
            return ctx
        
        # Perform validation
        calendar_violations = []
        butterfly_violations = []
        
        # Calendar arbitrage check
        for j in range(len(vol_surface.strikes)):
            for i in range(len(vol_surface.expiries) - 1):
                vol_short = vol_surface.vols[i, j]
                vol_long = vol_surface.vols[i + 1, j]
                
                if np.isnan(vol_short) or np.isnan(vol_long):
                    continue
                
                # Total variance should increase with expiry
                var_short = vol_short ** 2 * vol_surface.expiries[i]
                var_long = vol_long ** 2 * vol_surface.expiries[i + 1]
                
                if var_long < var_short - tolerance:
                    calendar_violations.append({
                        "strike": float(vol_surface.strikes[j]),
                        "expiry_short": float(vol_surface.expiries[i]),
                        "expiry_long": float(vol_surface.expiries[i + 1]),
                        "var_diff": float(var_long - var_short),
                    })
        
        # Butterfly arbitrage check (smile convexity)
        for i in range(len(vol_surface.expiries)):
            for j in range(1, len(vol_surface.strikes) - 1):
                vol_left = vol_surface.vols[i, j - 1]
                vol_mid = vol_surface.vols[i, j]
                vol_right = vol_surface.vols[i, j + 1]
                
                if np.isnan(vol_left) or np.isnan(vol_mid) or np.isnan(vol_right):
                    continue
                
                # Second derivative should be non-negative (convex smile)
                curvature = vol_left - 2 * vol_mid + vol_right
                if curvature < -tolerance:
                    butterfly_violations.append({
                        "expiry": float(vol_surface.expiries[i]),
                        "strike": float(vol_surface.strikes[j]),
                        "curvature": float(curvature),
                    })
        
        # Build report
        report = {
            "calendar_violations": len(calendar_violations),
            "butterfly_violations": len(butterfly_violations),
            "calendar_details": calendar_violations[:5],  # First 5
            "butterfly_details": butterfly_violations[:5],
            "passed": len(calendar_violations) == 0 and len(butterfly_violations) == 0,
        }
        
        ctx.put(Keys.ARBITRAGE_REPORT, report)
        
        if ctx.logger:
            if report["passed"]:
                ctx.logger.info("Arbitrage validation passed: no violations detected")
            else:
                ctx.logger.warning(
                    "Arbitrage violations: %d calendar, %d butterfly",
                    len(calendar_violations), len(butterfly_violations)
                )
        
        return ctx


@dataclass(slots=True)
class InterpolateSurfaceStep(Step):
    """
    Step 5: Apply surface interpolation.
    
    Interpolates missing vol points using configured method
    (SABR, SVI, bilinear, etc.).
    
    Outputs
    -------
    Updates ctx.state[Keys.VOL_SURFACE] with interpolated surface.
    """
    
    def run(self, ctx: Context) -> Context:
        # Get surface and config
        vol_surface: GridVolSurface = ctx.get(Keys.VOL_SURFACE)
        vol_cfg = _vol_surface_cfg(ctx.cfg)
        
        # Get interpolation config
        interp_cfg = vol_cfg.get("interpolation", {})
        strike_interp = interp_cfg.get("strike", "linear") if isinstance(interp_cfg, dict) else "linear"
        time_interp = interp_cfg.get("time", "linear_variance") if isinstance(interp_cfg, dict) else "linear"
        
        # Fill NaN values with interpolation
        filled_vols = self._interpolate_grid(
            vol_surface.vols.copy(),
            vol_surface.expiries,
            vol_surface.strikes,
            strike_interp,
            time_interp,
        )
        
        # Create interpolated surface
        interpolated_surface = GridVolSurface(
            expiries=vol_surface.expiries,
            strikes=vol_surface.strikes,
            vols=filled_vols,
            extrapolation="flat",
        )
        
        # Update state
        ctx.put(Keys.VOL_SURFACE, interpolated_surface)
        
        if ctx.logger:
            nan_count_before = np.sum(np.isnan(vol_surface.vols))
            nan_count_after = np.sum(np.isnan(filled_vols))
            ctx.logger.info(
                "Interpolated surface: filled %d missing points (strike=%s, time=%s)",
                nan_count_before - nan_count_after, strike_interp, time_interp
            )
        
        return ctx
    
    def _interpolate_grid(
        self,
        vols: np.ndarray,
        expiries: np.ndarray,
        strikes: np.ndarray,
        strike_interp: str,
        time_interp: str,
    ) -> np.ndarray:
        """
        Interpolate missing values in vol grid.
        """
        from scipy.interpolate import interp1d
        
        n_exp, n_strike = vols.shape
        
        # First interpolate along strike dimension for each expiry
        for i in range(n_exp):
            row = vols[i, :]
            valid_mask = ~np.isnan(row)
            if np.sum(valid_mask) >= 2:
                valid_strikes = strikes[valid_mask]
                valid_vols = row[valid_mask]
                
                interp_fn = interp1d(
                    valid_strikes, valid_vols,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(valid_vols[0], valid_vols[-1]),
                )
                vols[i, :] = interp_fn(strikes)
        
        # Then interpolate along time dimension for each strike
        for j in range(n_strike):
            col = vols[:, j]
            valid_mask = ~np.isnan(col)
            if np.sum(valid_mask) >= 2:
                valid_expiries = expiries[valid_mask]
                valid_vols = col[valid_mask]
                
                if time_interp == "linear_variance":
                    # Interpolate total variance, then convert back
                    valid_vars = valid_vols ** 2 * valid_expiries
                    interp_fn = interp1d(
                        valid_expiries, valid_vars,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(valid_vars[0], valid_vars[-1]),
                    )
                    total_vars = interp_fn(expiries)
                    vols[:, j] = np.sqrt(total_vars / np.maximum(expiries, 1e-6))
                else:
                    interp_fn = interp1d(
                        valid_expiries, valid_vols,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(valid_vols[0], valid_vols[-1]),
                    )
                    vols[:, j] = interp_fn(expiries)
        
        return vols


@dataclass(slots=True)
class StoreSurfaceStep(Step):
    """
    Step 6: Store surface in artifacts and log summary.
    
    Writes the calibrated surface to artifact store.
    
    Outputs
    -------
    Artifact file written to store.
    """
    
    def run(self, ctx: Context) -> Context:
        # Get surface and config
        vol_surface: GridVolSurface = ctx.get(Keys.VOL_SURFACE)
        vol_cfg = _vol_surface_cfg(ctx.cfg)
        
        underlying = str(vol_cfg.get("underlying", "UNKNOWN"))
        
        if ctx.logger:
            # Log sample vols
            mid_exp_idx = len(vol_surface.expiries) // 2
            mid_strike_idx = len(vol_surface.strikes) // 2
            sample_vol = vol_surface.vols[mid_exp_idx, mid_strike_idx]
            
            ctx.logger.info(
                "Vol surface %s complete: sample vol at T=%.2f, K=%.2f is %.2f%%",
                underlying,
                vol_surface.expiries[mid_exp_idx],
                vol_surface.strikes[mid_strike_idx],
                sample_vol * 100,
            )
        
        # Write to artifact store
        if ctx.artifact_store:
            surface_data = {
                "underlying": underlying,
                "expiries": list(vol_surface.expiries),
                "strikes": list(vol_surface.strikes),
                "vols": vol_surface.vols.tolist(),
            }
            
            import json
            artifact_path = ctx.artifact_store.artifacts_root / f"vol_surface_{underlying.lower()}.json"
            with open(artifact_path, "w") as f:
                json.dump(surface_data, f, indent=2)
        
        return ctx


# =============================================================================
# Pipeline Builder
# =============================================================================

def build_pipeline(cfg: RunConfig) -> Pipeline:
    """
    Build the marketdata.build_vol_surface pipeline.
    
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
        LoadVolQuotesStep(name="load_vol_quotes"),         # Step 1
        ConvertQuotesStep(name="convert_quotes"),          # Step 2
        BuildRawSurfaceStep(name="build_raw_surface"),     # Step 3
        ValidateArbitrageStep(name="validate_arbitrage"),  # Step 4
        InterpolateSurfaceStep(name="interpolate_surface"),  # Step 5
        StoreSurfaceStep(name="store_surface"),            # Step 6
    ]
    
    return Pipeline(
        name="marketdata.build_vol_surface",
        steps=steps,
    )
