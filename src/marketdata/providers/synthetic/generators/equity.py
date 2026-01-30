"""
Equity Synthetic Market Data Generator.

This module provides synthetic market data generators for equity instruments,
following the same patterns as FX generators but adapted for equity-specific
conventions:

    - Strike-based vol surfaces (not delta-based like FX)
    - Dividend yield handling
    - Single discount curve (not dom/for like FX)

Generators Provided
-------------------
- EQUITY SPOT: GBM spot paths with optional dividend yield in drift
- EQUITY FIXING: Reuse SPOT if present, else conservative constant
- EQUITY VOL: Strike-based implied volatility surface

Author: QuantStrata Team
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.specs import SpotGbmSpec, VolGridSmileSpec
from src.marketdata.surfaces.factory import GridVolFactory
from src.marketdata.providers.synthetic.context import SyntheticGenerationState
from src.marketdata.providers.synthetic.engine import rng_for_market_id
from src.marketdata.providers.synthetic.registry import SyntheticRegistry


def register_equity_generators(
    *,
    registry: SyntheticRegistry,
    base_seed: int,
    config: SyntheticProviderConfig,
) -> None:
    """
    Register equity generators into the synthetic registry.

    Generators Registered
    ---------------------
    - EQUITY SPOT: GBM spot paths with dividend yield
    - EQUITY FIXING: Reuse SPOT if present, else conservative constant
    - EQUITY VOL: Strike-based smile surface (depends on SPOT + discount curve)

    Parameters
    ----------
    registry : SyntheticRegistry
        Registry to add generators to.
    base_seed : int
        Base seed for deterministic RNG derivation.
    config : SyntheticProviderConfig
        Configuration object providing per-id spec overrides.
    """
    # Create the equity generator bundle object.
    eq = _EquityGenerators(base_seed=base_seed, config=config)

    # Register EQUITY SPOT generator.
    registry.register(
        asset_class="EQUITY",
        mkt_type="SPOT",
        generator=eq.generate_spot,
    )

    # Register EQUITY FIXING generator.
    registry.register(
        asset_class="EQUITY",
        mkt_type="FIXING",
        generator=eq.generate_fixing,
    )

    # Register EQUITY VOL generator with dependency requirements.
    registry.register(
        asset_class="EQUITY",
        mkt_type="VOL",
        generator=eq.generate_vol_grid_strike_based,
        requirements=eq.requirements_for_vol,
    )


@dataclass(frozen=True, slots=True)
class _EquityGenerators:
    """
    Equity synthetic generation container.

    Design Goals
    ------------
    - Deterministic and stable for tests/examples
    - Market-like dynamics:
        * Smile defined in strike space (not delta-based like FX)
        * GBM with dividend yield in drift
    - Clean fallback behavior if curves are missing

    Attributes
    ----------
    base_seed : int
        Seed used to derive deterministic RNG per MarketId.
    config : SyntheticProviderConfig
        Config object providing per-id spec overrides and defaults.
    """
    base_seed: int
    config: SyntheticProviderConfig

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    def requirements_for_vol(self, market_id: MarketId) -> Tuple[MarketId, ...]:
        """
        Return prerequisite MarketIds for EQUITY VOL generation.

        Requirements
        ------------
        - Corresponding EQUITY SPOT for the same ticker
        - Discount curve if currency qualifier is provided

        Parameters
        ----------
        market_id : MarketId
            The vol surface MarketId to check requirements for.

        Returns
        -------
        tuple[MarketId, ...]
            Required MarketIds for generating this vol surface.
        """
        # Construct the corresponding SPOT MarketId.
        spot_id = MarketId(
            asset_class=market_id.asset_class,  # "EQUITY"
            mkt_type="SPOT",
            name=market_id.name,  # Ticker (e.g., "AAPL", "SPX")
            qualifiers=market_id.qualifiers,
        )

        # Start with SPOT requirement.
        reqs = [spot_id]

        # Extract currency from qualifiers if present.
        ccy = _equity_currency(market_id)

        # Add discount curve if currency is provided.
        if ccy is not None:
            reqs.append(_default_ir_curve_id(ccy=ccy))

        return tuple(reqs)

    # ------------------------------------------------------------------
    # SPOT (GBM with dividend yield)
    # ------------------------------------------------------------------

    def generate_spot(
        self,
        market_id: MarketId,
        state: SyntheticGenerationState,
    ) -> None:
        """
        Generate EQUITY SPOT as a GBM panel [T, S].

        The drift includes dividend yield adjustment:
            dS = (μ - q) S dt + σ S dW

        where q is the continuous dividend yield.

        Storage
        -------
        - quote_panels[mid] = Panel(data=[T,S], axis_names=("time","scenario"))
        - spot_cache[mid] = raw np.ndarray for FIXING reuse
        """
        # Create deterministic RNG stream for this MarketId.
        rng = rng_for_market_id(base_seed=int(self.base_seed), market_id=market_id)

        # Pull spot generation spec (allows per-MarketId overrides).
        spec: SpotGbmSpec = self.config.spot_spec(market_id)

        # Generate the GBM spot paths as a dense array [T, S].
        spot = _generate_gbm_spot_panel(
            rng=rng,
            n_time=int(state.n_time),
            n_scenarios=int(state.n_scenarios),
            initial_level=float(spec.initial_level),
            drift=float(spec.drift),
            vol=float(spec.vol),
            dt=float(spec.dt),
            initial_dispersion=float(spec.initial_dispersion),
        )

        # Store in quote panels with declared axes.
        state.quote_panels[market_id] = Panel(
            data=spot,
            axis_names=("time", "scenario"),
        )
        # Cache raw spot array so FIXING can reuse exactly.
        state.spot_cache[market_id] = spot

    # ------------------------------------------------------------------
    # FIXING (reuse SPOT if available)
    # ------------------------------------------------------------------

    def generate_fixing(
        self,
        market_id: MarketId,
        state: SyntheticGenerationState,
    ) -> None:
        """
        Generate EQUITY FIXING.

        Policy
        ------
        - Reuse the corresponding SPOT if already generated
        - Else, create a conservative constant time-series

        Parameters
        ----------
        market_id : MarketId
            The FIXING MarketId to generate.
        state : SyntheticGenerationState
            Shared generation state.
        """
        # Build the corresponding SPOT MarketId.
        spot_id = MarketId(
            asset_class=market_id.asset_class,  # "EQUITY"
            mkt_type="SPOT",
            name=market_id.name,  # Same ticker
            qualifiers=market_id.qualifiers,
        )

        # If spot exists in cache, reuse it exactly.
        if spot_id in state.spot_cache:
            reused = state.spot_cache[spot_id]
            state.quote_panels[market_id] = Panel(
                data=reused,
                axis_names=("time", "scenario"),
            )
            return

        # If spot does not exist, fall back to a constant series.
        constant = np.full(
            (int(state.n_time), int(state.n_scenarios)),
            100.0,  # Default equity level
            dtype=float,
        )
        state.quote_panels[market_id] = Panel(
            data=constant,
            axis_names=("time", "scenario"),
        )

    # ------------------------------------------------------------------
    # VOL (strike-based smile)
    # ------------------------------------------------------------------

    def generate_vol_grid_strike_based(
        self,
        market_id: MarketId,
        state: SyntheticGenerationState,
    ) -> None:
        """
        Generate EQUITY VOL surface on a grid [T, S, n_exp, n_k] in absolute strike.

        Unlike FX (which uses forward-moneyness), equity surfaces are typically
        quoted directly in strike space with a skew/smile parameterization:

            σ(T, K) = ATM(T) × (1 + skew × m + smile × m²)

        where m = (K - S₀) / S₀ is spot moneyness (centered at current spot).

        Storage
        -------
        - vol_param_panels[mid] = Panel([T,S,n_exp,n_k], axes=("time","scenario","expiry","strike"))
        - vol_factories[mid] = GridVolFactory(expiries=..., strikes=...)
        """
        # Create deterministic RNG stream for this vol MarketId.
        rng = rng_for_market_id(base_seed=int(self.base_seed), market_id=market_id)

        # Pull vol surface generation spec (allows per-id overrides).
        spec: VolGridSmileSpec = self.config.vol_spec(market_id)

        # Normalize expiries/strikes to 1D float arrays.
        expiries = np.asarray(spec.expiries, dtype=float).reshape(-1)
        strikes = np.asarray(spec.strikes, dtype=float).reshape(-1)

        # Validate non-empty grids.
        if expiries.size == 0 or strikes.size == 0:
            raise ValueError("VolGridSmileSpec.expiries/strikes must be non-empty.")

        # --- Resolve dependencies (spot) ---
        spot_id = MarketId(
            asset_class=market_id.asset_class,
            mkt_type="SPOT",
            name=market_id.name,
            qualifiers=market_id.qualifiers,
        )

        # Fetch the spot panel produced earlier.
        spot_panel = state.quote_panels.get(spot_id)

        if spot_panel is None:
            raise ValueError(
                "EQUITY VOL generation requires SPOT to be present.\n"
                f"  missing spot_id={spot_id.key()}\n"
                f"  requested vol_id={market_id.key()}"
            )

        # --- Generate the full cube ---
        vol_cube = _generate_equity_vol_cube_strike_based(
            rng=rng,
            spot_panel=np.asarray(spot_panel.data, dtype=float),
            expiries=expiries,
            strikes=strikes,
            atm_vol=float(spec.atm_vol),
            skew=float(spec.skew),
            smile=float(spec.smile),
            term=float(spec.term),
            noise_scale=float(spec.noise_scale),
        )

        # Store the vol cube with semantic axis names.
        state.vol_param_panels[market_id] = Panel(
            data=vol_cube,
            axis_names=("time", "scenario", "expiry", "strike"),
        )

        # Store the factory for reconstructing a GridVolSurface at snapshot time.
        state.vol_factories[market_id] = GridVolFactory(
            expiries=expiries,
            strikes=strikes,
            extrapolation=str(spec.extrapolation),
        )


# -------------------------------------------------------------------------
# Helpers: qualifiers + curve lookup
# -------------------------------------------------------------------------

def _iter_qualifier_items(qualifiers) -> list[tuple[str, str]]:
    """
    Normalize MarketId.qualifiers into a list of (key, value) pairs.

    Accepts:
    - None
    - tuple/list of pairs: (("k","v"), ...)
    - Mapping: {"k": "v", ...}

    Returns
    -------
    list[(str, str)]
        Normalized (key, value) pairs.
    """
    if qualifiers is None:
        return []

    # Mapping-like (dict) case.
    if hasattr(qualifiers, "items"):
        out: list[tuple[str, str]] = []
        for k, v in qualifiers.items():
            out.append((str(k), str(v)))
        return out

    # Tuple-of-pairs (or list-of-pairs) case.
    out = []
    try:
        for kv in qualifiers:
            k, v = kv
            out.append((str(k), str(v)))
        return out
    except Exception as e:
        raise TypeError(
            "MarketId.qualifiers must be None, a mapping, or an iterable of (key,value) pairs. "
            f"Got type={type(qualifiers)!r}, value={qualifiers!r}."
        ) from e


def _equity_currency(market_id: MarketId) -> Optional[str]:
    """
    Extract currency from MarketId qualifiers.

    Accepted qualifier keys (case-insensitive): ccy, currency

    Returns
    -------
    str or None
        Currency code (uppercase) if found, else None.
    """
    items = _iter_qualifier_items(market_id.qualifiers)

    for k, v in items:
        key_lower = str(k).strip().lower()
        if key_lower in ("ccy", "currency"):
            ccy = str(v).strip().upper()
            return ccy if ccy else None

    return None


def _default_ir_curve_id(ccy: str) -> MarketId:
    """
    Construct a canonical IR curve MarketId for a currency.

    Format: IR.CURVE.<CCY>.OIS|ccy=<CCY>

    Parameters
    ----------
    ccy : str
        Currency code (e.g., "USD").

    Returns
    -------
    MarketId
        Canonical IR curve MarketId.
    """
    ccy_u = str(ccy).strip().upper()

    return MarketId(
        asset_class="IR",
        mkt_type="CURVE",
        name=f"{ccy_u}.OIS",
        qualifiers=(("ccy", ccy_u),),
    )


# -------------------------------------------------------------------------
# Equity numeric generators
# -------------------------------------------------------------------------

def _generate_gbm_spot_panel(
    *,
    rng: np.random.Generator,
    n_time: int,
    n_scenarios: int,
    initial_level: float,
    drift: float,
    vol: float,
    dt: float,
    initial_dispersion: float,
) -> np.ndarray:
    """
    Generate GBM spot panel [T, S].

    Model
    -----
        S_{t+1} = S_t × exp((μ - ½σ²)dt + σ√dt × Z)

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_time : int
        Number of time points.
    n_scenarios : int
        Number of scenarios.
    initial_level : float
        Starting spot level (e.g., 100.0 for a stock).
    drift : float
        Drift parameter μ (can include dividend yield adjustment).
    vol : float
        Volatility parameter σ.
    dt : float
        Time step size in years.
    initial_dispersion : float
        Optional dispersion at t=0 (log-normal spread around initial_level).

    Returns
    -------
    np.ndarray
        Spot array of shape [n_time, n_scenarios].
    """
    # Validate inputs.
    if initial_level <= 0.0:
        raise ValueError("initial_level must be > 0.")
    if vol < 0.0:
        raise ValueError("vol must be >= 0.")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")

    t_count = int(n_time)
    s_count = int(n_scenarios)

    # Allocate output array [T, S].
    spot = np.empty((t_count, s_count), dtype=float)

    # Initial values across scenarios.
    if float(initial_dispersion) > 0.0:
        z0 = rng.normal(size=(s_count,))
        spot[0, :] = float(initial_level) * np.exp(float(initial_dispersion) * z0)
    else:
        spot[0, :] = float(initial_level)

    # If only one time point, return immediately.
    if t_count == 1:
        return spot

    # Draw standard normals for each (time step, scenario).
    z = rng.normal(size=(t_count - 1, s_count))

    mu = float(drift)
    sigma = float(vol)

    # Compute log-return increments.
    log_returns = (mu - 0.5 * sigma * sigma) * float(dt) + sigma * np.sqrt(float(dt)) * z

    # Apply recurrence.
    for t in range(1, t_count):
        spot[t, :] = spot[t - 1, :] * np.exp(log_returns[t - 1, :])

    return spot


def _generate_equity_vol_cube_strike_based(
    *,
    rng: np.random.Generator,
    spot_panel: np.ndarray,
    expiries: np.ndarray,
    strikes: np.ndarray,
    atm_vol: float,
    skew: float,
    smile: float,
    term: float,
    noise_scale: float,
) -> np.ndarray:
    """
    Generate vol cube [T, S, n_exp, n_k] using strike-based parameterization.

    Model (Equity Style)
    --------------------
    For each expiry T and strike K:
        m = (K - S₀) / S₀                    # Spot moneyness (centered)
        ATM(T) = atm_vol × (1 + term × (√T - √T_ref))  # Term structure
        σ(T, K) = ATM(T) × (1 + skew × m + smile × m²)

    This produces:
        - Negative skew for m < 0 (low strikes have higher vol)
        - Smile effect from the quadratic term

    Parameters
    ----------
    rng : np.random.Generator
        Random generator for optional noise.
    spot_panel : np.ndarray
        Spot array [T, S] from earlier generation.
    expiries : np.ndarray
        Expiry grid (year fractions).
    strikes : np.ndarray
        Strike grid (absolute strikes).
    atm_vol : float
        Base ATM volatility level.
    skew : float
        Linear moneyness coefficient (typically negative for equity).
    smile : float
        Quadratic moneyness coefficient (positive for smile convexity).
    term : float
        Term structure scaling factor.
    noise_scale : float
        Standard deviation of additive noise (0 for deterministic).

    Returns
    -------
    np.ndarray
        Vol cube of shape [n_time, n_scenarios, n_expiries, n_strikes].
    """
    # Validate spot panel.
    spot = np.asarray(spot_panel, dtype=float)
    if spot.ndim != 2:
        raise ValueError(f"spot_panel must be [T, S], got shape={spot.shape}.")

    t_count, s_count = int(spot.shape[0]), int(spot.shape[1])

    # Normalize grids.
    exp = np.asarray(expiries, dtype=float).reshape(-1)
    k = np.asarray(strikes, dtype=float).reshape(-1)

    n_exp = int(exp.size)
    n_k = int(k.size)

    # Allocate output cube [T, S, n_exp, n_k].
    out = np.empty((t_count, s_count, n_exp, n_k), dtype=float)

    # Reference expiry for term scaling.
    t_ref = float(exp[n_exp // 2])
    t_ref = max(t_ref, 1e-6)

    # Loop over all times and scenarios.
    for ti in range(t_count):
        for si in range(s_count):
            # Read spot for this (time, scenario).
            s0 = float(spot[ti, si])

            if not np.isfinite(s0) or s0 <= 0.0:
                raise ValueError(
                    f"Invalid spot at (time, scenario)=({ti}, {si}): S={s0}"
                )

            # Loop over expiries.
            for ei in range(n_exp):
                T = float(exp[ei])

                # Term structure on ATM vol.
                atm_T = float(atm_vol) * (
                    1.0 + float(term) * (np.sqrt(max(T, 1e-6)) - np.sqrt(t_ref))
                )
                atm_T = max(atm_T, 1e-4)  # Floor to avoid degenerate vols.

                # Spot moneyness: m = (K - S₀) / S₀.
                m = (k - s0) / max(s0, 1e-12)

                # Smile formula: σ = ATM(T) × (1 + skew × m + smile × m²).
                sigma = atm_T * (1.0 + float(skew) * m + float(smile) * (m ** 2))

                # Floor to keep positive.
                sigma = np.maximum(sigma, 1e-4)

                # Store the slice for this expiry.
                out[ti, si, ei, :] = sigma

    # Add noise if requested.
    if float(noise_scale) > 0.0:
        out = out + rng.normal(loc=0.0, scale=float(noise_scale), size=out.shape)

    # Final floor for positivity.
    return np.maximum(out, 1e-4)


# -------------------------------------------------------------------------
# Dividend Models (helper for discrete dividend adjustments)
# -------------------------------------------------------------------------

def adjust_spot_for_discrete_dividend(
    *,
    spot: float,
    dividend_amount: float,
    ex_date_fraction: float,
    current_time: float,
) -> float:
    """
    Adjust spot price for a discrete dividend payment.

    Model
    -----
    If the dividend ex-date is in the future (ex_date_fraction > current_time),
    the adjusted spot is:

        S_adj = S - D × exp(-r × (t_ex - t))

    where D is the dividend amount and t_ex is the ex-date.

    For simplicity, this implementation assumes zero discount rate (r=0),
    so:
        S_adj = S - D   (if ex-date in future)
        S_adj = S       (if ex-date has passed)

    Parameters
    ----------
    spot : float
        Current spot price.
    dividend_amount : float
        Absolute dividend amount per share.
    ex_date_fraction : float
        Ex-dividend date as year fraction.
    current_time : float
        Current time as year fraction.

    Returns
    -------
    float
        Dividend-adjusted spot price.

    Notes
    -----
    For more sophisticated handling with discounting, use:
        S_adj = S - PV(D) where PV(D) = D × exp(-r × (t_ex - t))
    """
    if ex_date_fraction > current_time:
        # Dividend is in the future, subtract it.
        adjusted = float(spot) - float(dividend_amount)
        return max(adjusted, 1e-6)  # Floor to avoid non-positive spot.
    else:
        # Dividend already paid, no adjustment.
        return float(spot)


def compute_forward_with_dividends(
    *,
    spot: float,
    discount_rate: float,
    dividend_yield: float,
    expiry: float,
    discrete_dividends: Optional[list[tuple[float, float]]] = None,
) -> float:
    """
    Compute equity forward price with continuous and/or discrete dividends.

    Model
    -----
    With continuous dividend yield q:
        F = S × exp((r - q) × T)

    With discrete dividends D_i at times t_i < T:
        F = (S - Σ D_i × exp(-r × t_i)) × exp(r × T)

    Combined model (continuous yield + discrete):
        F = (S - PV(discrete_divs)) × exp((r - q) × T)

    Parameters
    ----------
    spot : float
        Current spot price.
    discount_rate : float
        Continuous discount rate r.
    dividend_yield : float
        Continuous dividend yield q.
    expiry : float
        Time to expiry T in years.
    discrete_dividends : list[(t_i, D_i)] or None
        List of (ex_date, amount) pairs for discrete dividends.

    Returns
    -------
    float
        Forward price F.
    """
    r = float(discount_rate)
    q = float(dividend_yield)
    T = float(expiry)
    S = float(spot)

    # Start with spot.
    adjusted_spot = S

    # Subtract present value of discrete dividends.
    if discrete_dividends:
        for t_i, D_i in discrete_dividends:
            t_ex = float(t_i)
            div_amt = float(D_i)
            if 0.0 < t_ex < T:
                # PV of dividend = D × exp(-r × t_ex).
                pv_div = div_amt * np.exp(-r * t_ex)
                adjusted_spot -= pv_div

    # Floor to avoid non-positive.
    adjusted_spot = max(adjusted_spot, 1e-6)

    # Forward with continuous dividend yield.
    forward = adjusted_spot * np.exp((r - q) * T)

    return float(forward)


# -------------------------------------------------------------------------
# EXPORTS
# -------------------------------------------------------------------------

__all__ = [
    "register_equity_generators",
    "adjust_spot_for_discrete_dividend",
    "compute_forward_with_dividends",
]
