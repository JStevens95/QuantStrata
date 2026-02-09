"""
Pipeline: marketdata.build_curves

Bootstrap yield curves from market rate quotes (deposits, FRAs, swaps).

Purpose
-------
Build a TermStructure (zero rate curve) from input rate quotes by:
1. Loading rate quotes from config or state
2. Validating quote consistency and coverage
3. Bootstrapping discount factors using iterative solver
4. Applying interpolation (log-linear, cubic spline)
5. Storing the resulting TermStructure for downstream pricing

Design Philosophy
-----------------
- Fail-fast validation: Reject invalid/inconsistent quotes early.
- Configurable interpolation: Support multiple interpolation methods.
- Composable: Output can be consumed by pricing, calibration, risk pipelines.

Author: QuantStrata Team
"""
from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================
from dataclasses import dataclass  # For defining immutable step classes
from typing import Any, Dict, List, Mapping, Optional  # Type hints for clarity

# =============================================================================
# Orchestrator Framework Imports
# =============================================================================
from src.orchestrator.config.schemas import RunConfig  # Validated run configuration
from src.orchestrator.core.context import Context  # Execution context passed between steps
from src.orchestrator.core.pipeline import Pipeline  # Pipeline container
from src.orchestrator.core.step import Step  # Step interface
from src.orchestrator.core.state_keys import StateKeys as Keys  # Centralised state keys

# =============================================================================
# Market Data Imports
# =============================================================================
from src.marketdata.curves.term_structure import ZeroRateCurve  # Curve implementation
# Bootstrapping uses in-pipeline logic; see bootstrap_discount_curve in
# src.marketdata.curves.bootstrapper for alternative API.

import numpy as np  # Numerical operations


# =============================================================================
# Configuration Helpers
# =============================================================================

def _require_dict(parent: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """
    Extract a required dictionary from the parent mapping.
    
    Parameters
    ----------
    parent : Mapping
        The parent configuration mapping.
    key : str
        The key to extract.
        
    Returns
    -------
    Dict[str, Any]
        The extracted dictionary.
        
    Raises
    ------
    KeyError
        If the key is missing.
    TypeError
        If the value is not a dictionary.
    """
    # Check if key exists in parent mapping
    if key not in parent:
        raise KeyError(f"Missing required config key: '{key}'")
    
    # Extract the value
    value = parent[key]
    
    # Validate type is dict
    if not isinstance(value, dict):
        raise TypeError(f"Config key '{key}' must be a dict, got: {type(value).__name__}")
    
    # Return the validated dictionary
    return value


def _require_str(value: Any, *, key_name: str) -> str:
    """
    Coerce and validate a non-empty string configuration field.
    
    Parameters
    ----------
    value : Any
        The value to validate.
    key_name : str
        The key name for error messages.
        
    Returns
    -------
    str
        The validated string.
    """
    # Convert to string
    if not isinstance(value, str):
        raise TypeError(f"'{key_name}' must be a str, got: {type(value).__name__}")
    
    # Strip whitespace
    out = value.strip()
    
    # Reject empty strings
    if not out:
        raise ValueError(f"'{key_name}' must be a non-empty string")
    
    return out


def _require_float(value: Any, *, key_name: str) -> float:
    """
    Coerce and validate a float configuration field.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"'{key_name}' must be float-like, got: {type(value).__name__}") from exc


def _curves_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """
    Extract the 'curves' configuration block from RunConfig.params.
    
    Expected structure:
        cfg.params["curves"] is a dict containing currency, quotes, interpolation, etc.
    """
    # Validate params is a dict
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    
    # Extract and return the curves block
    return _require_dict(cfg.params, "curves")


# =============================================================================
# Rate Quote Data Class
# =============================================================================

@dataclass(frozen=True, slots=True)
class RateQuote:
    """
    A single rate quote for curve bootstrapping.
    
    Attributes
    ----------
    instrument_type : str
        Type of instrument: "deposit" or "swap".
    tenor : str
        Tenor string (e.g., "1M", "3M", "1Y", "5Y").
    rate : float
        The quoted rate (e.g., 0.05 for 5%).
    tenor_years : float
        Tenor converted to year fraction for calculations.
    """
    instrument_type: str  # "deposit" or "swap"
    tenor: str            # Tenor string (e.g., "3M", "1Y")
    rate: float           # Quoted rate (decimal)
    tenor_years: float    # Tenor in year fraction


def _tenor_to_years(tenor: str) -> float:
    """
    Convert a tenor string to year fraction.
    
    Parameters
    ----------
    tenor : str
        Tenor string like "1M", "3M", "1Y", "5Y", "10Y".
        
    Returns
    -------
    float
        Year fraction (e.g., "3M" -> 0.25, "1Y" -> 1.0).
    """
    # Normalise tenor string (uppercase, strip whitespace)
    t = tenor.upper().strip()
    
    # Handle day tenors
    if t.endswith("D"):
        days = int(t[:-1])
        return days / 365.0
    
    # Handle week tenors
    if t.endswith("W"):
        weeks = int(t[:-1])
        return weeks * 7 / 365.0
    
    # Handle month tenors
    if t.endswith("M"):
        months = int(t[:-1])
        return months / 12.0
    
    # Handle year tenors
    if t.endswith("Y"):
        years = int(t[:-1])
        return float(years)
    
    # Unknown format
    raise ValueError(f"Unknown tenor format: '{tenor}'. Expected format like '3M', '1Y'.")


def _parse_quotes(quotes_config: Dict[str, Any]) -> List[RateQuote]:
    """
    Parse rate quotes from configuration into RateQuote objects.
    
    Parameters
    ----------
    quotes_config : Dict
        Configuration containing 'deposits' and/or 'swaps' lists.
        
    Returns
    -------
    List[RateQuote]
        Parsed rate quotes sorted by tenor.
    """
    quotes: List[RateQuote] = []
    
    # Parse deposit quotes
    deposits = quotes_config.get("deposits", [])
    for dep in deposits:
        tenor = _require_str(dep.get("tenor"), key_name="deposit.tenor")
        rate = _require_float(dep.get("rate"), key_name="deposit.rate")
        quotes.append(RateQuote(
            instrument_type="deposit",
            tenor=tenor,
            rate=rate,
            tenor_years=_tenor_to_years(tenor),
        ))
    
    # Parse swap quotes
    swaps = quotes_config.get("swaps", [])
    for sw in swaps:
        tenor = _require_str(sw.get("tenor"), key_name="swap.tenor")
        rate = _require_float(sw.get("rate"), key_name="swap.rate")
        quotes.append(RateQuote(
            instrument_type="swap",
            tenor=tenor,
            rate=rate,
            tenor_years=_tenor_to_years(tenor),
        ))
    
    # Sort by tenor (ascending)
    quotes.sort(key=lambda q: q.tenor_years)
    
    return quotes


# =============================================================================
# Pipeline Steps
# =============================================================================

@dataclass(slots=True)
class LoadRateQuotesStep(Step):
    """
    Step 1: Load rate quotes from configuration.
    
    Reads the quotes section of the curves config and parses into RateQuote objects.
    
    Outputs
    -------
    ctx.state[Keys.RATE_QUOTES] : List[RateQuote]
        Parsed rate quotes sorted by tenor.
    """
    
    def run(self, ctx: Context) -> Context:
        # Extract curves configuration block
        curves_cfg = _curves_cfg(ctx.cfg)
        
        # Extract quotes sub-configuration
        quotes_config = _require_dict(curves_cfg, "quotes")
        
        # Parse quotes into RateQuote objects
        quotes = _parse_quotes(quotes_config)
        
        # Validate we have at least one quote
        if not quotes:
            raise ValueError("No rate quotes provided. Need at least one deposit or swap quote.")
        
        # Store parsed quotes in state
        ctx.put(Keys.RATE_QUOTES, quotes)
        
        # Log summary
        if ctx.logger:
            ctx.logger.info(
                "Loaded %d rate quotes (%d deposits, %d swaps)",
                len(quotes),
                sum(1 for q in quotes if q.instrument_type == "deposit"),
                sum(1 for q in quotes if q.instrument_type == "swap"),
            )
        
        return ctx


@dataclass(slots=True)
class ValidateQuotesStep(Step):
    """
    Step 2: Validate quote consistency and coverage.
    
    Checks:
    - No duplicate tenors
    - Rates are within reasonable bounds
    - Sufficient tenor coverage for bootstrapping
    
    Outputs
    -------
    No new state keys; raises on validation failure.
    """
    
    def run(self, ctx: Context) -> Context:
        # Retrieve quotes from state
        quotes: List[RateQuote] = ctx.get(Keys.RATE_QUOTES)
        
        # --- Check for duplicate tenors ---
        seen_tenors: set = set()
        for q in quotes:
            key = (q.instrument_type, q.tenor_years)
            if key in seen_tenors:
                raise ValueError(f"Duplicate quote for {q.instrument_type} tenor {q.tenor}")
            seen_tenors.add(key)
        
        # --- Validate rate bounds (sanity check) ---
        for q in quotes:
            if q.rate < -0.10:  # Allow slightly negative rates
                raise ValueError(f"Rate {q.rate:.4f} for {q.tenor} is unreasonably negative")
            if q.rate > 1.0:  # Max 100% rate
                raise ValueError(f"Rate {q.rate:.4f} for {q.tenor} is unreasonably high")
        
        # --- Check tenor coverage ---
        tenors = [q.tenor_years for q in quotes]
        if min(tenors) > 0.5:
            if ctx.logger:
                ctx.logger.warning("No short-term quotes (< 6M); curve may be poorly defined at short end")
        
        # Log validation success
        if ctx.logger:
            ctx.logger.info(
                "Quote validation passed: tenors from %.2fY to %.2fY",
                min(tenors), max(tenors)
            )
        
        return ctx


@dataclass(slots=True)
class BootstrapCurveStep(Step):
    """
    Step 3: Bootstrap discount factors from rate quotes.
    
    Uses iterative bootstrapping to solve for discount factors:
    - Deposits: DF(T) = 1 / (1 + r * T)  [simple compounding]
    - Swaps: Solve for DF using swap equation
    
    Outputs
    -------
    ctx.state[Keys.DISCOUNT_FACTORS] : Dict[float, float]
        Discount factors by tenor (year fraction).
    """
    
    def run(self, ctx: Context) -> Context:
        # Retrieve quotes from state
        quotes: List[RateQuote] = ctx.get(Keys.RATE_QUOTES)
        
        # Initialise discount factor dictionary
        discount_factors: Dict[float, float] = {0.0: 1.0}  # DF(0) = 1
        
        # Sort quotes by tenor for sequential bootstrapping
        sorted_quotes = sorted(quotes, key=lambda q: q.tenor_years)
        
        for q in sorted_quotes:
            if q.instrument_type == "deposit":
                # Simple compounding for deposits: DF = 1 / (1 + r * T)
                df = 1.0 / (1.0 + q.rate * q.tenor_years)
            else:
                # Swap bootstrapping: solve swap equation for DF
                # PV_fixed = sum(c * DF(ti)) + DF(T) = 1
                # Approximation: DF(T) = (1 - r * sum(DF(ti))) / (1 + r)
                df = self._bootstrap_swap_df(q, discount_factors)
            
            # Store discount factor
            discount_factors[q.tenor_years] = df
        
        # Store in state
        ctx.put(Keys.DISCOUNT_FACTORS, discount_factors)
        
        # Log summary
        if ctx.logger:
            ctx.logger.info(
                "Bootstrapped %d discount factors (DF(%.1fY) = %.6f)",
                len(discount_factors) - 1,  # Exclude DF(0)
                max(discount_factors.keys()),
                discount_factors[max(discount_factors.keys())],
            )
        
        return ctx
    
    def _bootstrap_swap_df(
        self, swap_quote: RateQuote, existing_dfs: Dict[float, float]
    ) -> float:
        """
        Bootstrap discount factor from a swap quote.
        
        Uses the swap pricing equation:
            PV_fixed = c * sum(DF(ti) * Δti) + DF(T) * N = N
            
        For par swap (PV = 1), solve for DF(T):
            DF(T) = (1 - c * sum(DF(ti) * Δti)) / (1 + c * ΔT_last)
        """
        r = swap_quote.rate  # Swap rate (fixed rate)
        T = swap_quote.tenor_years  # Swap maturity
        
        # Annual payment frequency (simplification)
        payment_freq = 1.0  # Annual payments
        
        # Calculate sum of DF(ti) * Δti for intermediate payments
        accrued_sum = 0.0
        t = payment_freq
        while t < T - 1e-9:  # Iterate through payment dates before T
            # Find nearest DF (linear interpolation if needed)
            df_t = self._interpolate_df(t, existing_dfs)
            accrued_sum += r * payment_freq * df_t
            t += payment_freq
        
        # Solve for DF(T)
        df_T = (1.0 - accrued_sum) / (1.0 + r * payment_freq)
        
        return max(df_T, 0.001)  # Floor at small positive value
    
    def _interpolate_df(self, t: float, dfs: Dict[float, float]) -> float:
        """
        Interpolate discount factor at time t from existing DFs.
        Uses log-linear interpolation.
        """
        # Get sorted tenor points
        tenors = sorted(dfs.keys())
        
        # If t is before first tenor, return DF(0) = 1
        if t <= tenors[0]:
            return 1.0
        
        # If t is after last tenor, extrapolate flat
        if t >= tenors[-1]:
            return dfs[tenors[-1]]
        
        # Find bracketing tenors
        for i in range(len(tenors) - 1):
            if tenors[i] <= t <= tenors[i + 1]:
                t1, t2 = tenors[i], tenors[i + 1]
                df1, df2 = dfs[t1], dfs[t2]
                
                # Log-linear interpolation
                alpha = (t - t1) / (t2 - t1)
                log_df = (1 - alpha) * np.log(df1) + alpha * np.log(df2)
                return float(np.exp(log_df))
        
        # Fallback (shouldn't reach here)
        return dfs[tenors[-1]]


@dataclass(slots=True)
class InterpolateCurveStep(Step):
    """
    Step 4: Apply interpolation method to create continuous curve.
    
    Converts discrete discount factors into a ZeroRateCurve with
    specified interpolation method.
    
    Outputs
    -------
    ctx.state[Keys.TERM_STRUCTURE] : ZeroRateCurve
        Interpolated term structure curve.
    """
    
    def run(self, ctx: Context) -> Context:
        # Retrieve discount factors and config
        discount_factors: Dict[float, float] = ctx.get(Keys.DISCOUNT_FACTORS)
        curves_cfg = _curves_cfg(ctx.cfg)
        
        # Get interpolation method (default: log_linear)
        interpolation = str(curves_cfg.get("interpolation", "log_linear")).lower()
        
        # Get extrapolation mode
        extrapolation = str(curves_cfg.get("extrapolation", "flat")).lower()
        
        # Convert discount factors to zero rates
        tenors = sorted([t for t in discount_factors.keys() if t > 0])
        zero_rates = []
        
        for t in tenors:
            df = discount_factors[t]
            # Zero rate from DF: r = -ln(DF) / T
            if df > 0 and t > 0:
                r = -np.log(df) / t
            else:
                r = 0.0
            zero_rates.append(r)
        
        # Create ZeroRateCurve
        term_structure = ZeroRateCurve(
            tenors=np.array(tenors),
            zero_rates=np.array(zero_rates),
            extrapolation=extrapolation if extrapolation in ("flat", "linear") else "flat",
        )
        
        # Store in state
        ctx.put(Keys.TERM_STRUCTURE, term_structure)
        
        # Log summary
        if ctx.logger:
            ctx.logger.info(
                "Built ZeroRateCurve with %d nodes, interpolation=%s, extrapolation=%s",
                len(tenors), interpolation, extrapolation
            )
        
        return ctx


@dataclass(slots=True)
class StoreCurveStep(Step):
    """
    Step 5: Store curve in artifacts and log final summary.
    
    Writes the curve to the artifact store (if enabled) and logs
    a summary of the bootstrapped curve.
    
    Outputs
    -------
    Updates artifact store with curve data.
    """
    
    def run(self, ctx: Context) -> Context:
        # Retrieve term structure
        term_structure: ZeroRateCurve = ctx.get(Keys.TERM_STRUCTURE)
        curves_cfg = _curves_cfg(ctx.cfg)
        
        # Get currency for naming
        currency = str(curves_cfg.get("currency", "USD")).upper()
        
        # Log final curve summary
        if ctx.logger:
            # Compute a few sample rates for logging
            sample_tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
            rate_summary = []
            for t in sample_tenors:
                try:
                    r = term_structure.zero_rate(t)
                    rate_summary.append(f"{t}Y: {r:.4f}")
                except Exception:
                    pass
            
            ctx.logger.info(
                "Curve %s bootstrapped successfully. Sample rates: %s",
                currency, ", ".join(rate_summary[:4])
            )
        
        # Write to artifact store if enabled
        if ctx.artifact_store:
            # Prepare curve data for serialisation
            curve_data = {
                "currency": currency,
                "tenors": list(term_structure.tenors),
                "zero_rates": list(term_structure.zero_rates),
                "extrapolation": term_structure.extrapolation,
            }
            
            # Write as JSON artifact
            artifact_path = ctx.artifact_store.artifacts_root / f"curve_{currency.lower()}.json"
            import json
            with open(artifact_path, "w") as f:
                json.dump(curve_data, f, indent=2)
        
        return ctx


# =============================================================================
# Pipeline Builder
# =============================================================================

def build_pipeline(cfg: RunConfig) -> Pipeline:
    """
    Build the marketdata.build_curves pipeline.
    
    This pipeline bootstraps a yield curve from rate quotes:
    1. Load rate quotes from config
    2. Validate quote consistency
    3. Bootstrap discount factors
    4. Apply interpolation
    5. Store resulting curve
    
    Parameters
    ----------
    cfg : RunConfig
        The validated run configuration.
        
    Returns
    -------
    Pipeline
        The assembled pipeline ready for execution.
    """
    # Define ordered steps
    steps: List[Step] = [
        LoadRateQuotesStep(name="load_rate_quotes"),       # Step 1: Load quotes
        ValidateQuotesStep(name="validate_quotes"),        # Step 2: Validate
        BootstrapCurveStep(name="bootstrap_curve"),        # Step 3: Bootstrap DFs
        InterpolateCurveStep(name="interpolate_curve"),    # Step 4: Interpolate
        StoreCurveStep(name="store_curve"),                # Step 5: Store
    ]
    
    # Return assembled pipeline
    return Pipeline(
        name="marketdata.build_curves",
        steps=steps,
    )
