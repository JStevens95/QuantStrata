from __future__ import annotations

# NumPy is used for numeric arrays and vectorized math.
import numpy as np

# dataclass provides a clean immutable generator container.
from dataclasses import dataclass

# Tuple is used for typed multi-return functions (dom_ccy, for_ccy).
from typing import Tuple

# MarketId is the canonical identifier for market data objects.
from src.marketdata.core.ids import MarketId

# Panel wraps arrays with axis names.
from src.marketdata.core.panel import Panel

# Provider config + specs define how to parameterize synthetic generation.
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.specs import SpotGbmSpec, VolGridSmileSpec

# GridVolFactory reconstructs a pricing vol surface at snapshot time.
from src.marketdata.surfaces.factories import GridVolFactory

# Generation state is the shared mutable state mutated by generators.
from src.marketdata.providers.synthetic.context import SyntheticGenerationState

# rng_for_market_id derives deterministic RNG streams per MarketId.
from src.marketdata.providers.synthetic.engine import rng_for_market_id

# Registry is used to register these generators with the engine.
from src.marketdata.providers.synthetic.registry import SyntheticRegistry


def register_fx_generators(*, registry: SyntheticRegistry, base_seed: int, config: SyntheticProviderConfig) -> None:
    """
    Register FX generators into the synthetic registry.

    Generators registered
    ---------------------
    - FX SPOT   : GBM spot paths
    - FX FIXING : reuse SPOT if present, else conservative constant
    - FX VOL    : forward-moneyness smile surface (depends on SPOT + curves)
    """
    # Create the FX generator bundle object (holds config + base seed).
    fx = _FxGenerators(base_seed=base_seed, config=config)

    # Register FX SPOT generator under (asset_class="FX", mkt_type="SPOT").
    registry.register(asset_class="FX", mkt_type="SPOT", generator=fx.generate_spot)

    # Register FX FIXING generator under (asset_class="FX", mkt_type="FIXING").
    registry.register(asset_class="FX", mkt_type="FIXING", generator=fx.generate_fixing)

    # Register FX VOL generator under (asset_class="FX", mkt_type="VOL"),
    # and attach requirements resolver so the engine can auto-add SPOT/curves.
    registry.register(
        asset_class="FX",
        mkt_type="VOL",
        generator=fx.generate_vol_grid_forward_moneyness,
        requirements=fx.requirements_for_vol,
    )


@dataclass(frozen=True, slots=True)
class _FxGenerators:
    """
    FX synthetic generation container.

    Design goals
    ------------
    - Deterministic and stable for tests/examples
    - Market-like dynamics:
        * smile defined in log-forward-moneyness
        * forward built from spot and dom/for curves (carry)
    - Clean fallback behavior if curves are missing
    """
    # Seed used to derive deterministic RNG per MarketId.
    base_seed: int
    # Config object that provides per-id spec overrides and defaults.
    config: SyntheticProviderConfig

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    def requirements_for_vol(self, market_id: MarketId) -> Tuple[MarketId, ...]:
        """
        Return prerequisite MarketIds for FX VOL generation.

        We require:
        - corresponding FX SPOT for the same pair
        - domestic and foreign IR curves if qualifiers identify the currencies

        Notes
        -----
        This enables the engine to auto-expand the Universe (dependency closure).
        """
        # Construct the MarketId for the spot corresponding to this vol MarketId.
        spot_id = MarketId(
            asset_class=market_id.asset_class,     # Same asset class ("FX").
            mkt_type="SPOT",                       # Spot market type.
            name=market_id.name,                   # Same instrument name (e.g. "EURUSD").
            qualifiers=market_id.qualifiers,       # Same qualifiers (cut, convention, dom/for ccy, etc.).
        )

        # Extract domestic/foreign currencies from qualifiers (if present).
        dom_ccy, for_ccy = _fx_dom_for_ccy(market_id)

        # Start prerequisites with SPOT (always required for forward-moneyness vols).
        reqs = [spot_id]

        # Add domestic curve if domestic currency is provided.
        if dom_ccy is not None:
            reqs.append(_default_ir_curve_id(ccy=dom_ccy))

        # Add foreign curve if foreign currency is provided.
        if for_ccy is not None:
            reqs.append(_default_ir_curve_id(ccy=for_ccy))

        # Return as tuple for immutability / stable typing.
        return tuple(reqs)

    # ------------------------------------------------------------------
    # SPOT (GBM)
    # ------------------------------------------------------------------

    def generate_spot(self, market_id: MarketId, state: SyntheticGenerationState) -> None:
        """
        Generate FX SPOT as a GBM panel [T,S].

        Storage
        -------
        - quote_panels[mid] = Panel(data=[T,S], axis_names=("time","scenario"))
        - spot_cache[mid] = raw np.ndarray for FIXING reuse
        """
        # Create deterministic RNG stream for this MarketId.
        rng = rng_for_market_id(base_seed=int(self.base_seed), market_id=market_id)

        # Pull spot generation spec (allows per-MarketId overrides).
        spec: SpotGbmSpec = self.config.spot_spec(market_id)

        # Generate the GBM spot paths as a dense array [T,S].
        spot = _generate_gbm_spot_panel(
            rng=rng,                                 # RNG stream for this MarketId.
            n_time=int(state.n_time),                 # Number of time points.
            n_scenarios=int(state.n_scenarios),       # Number of scenarios.
            initial_level=float(spec.initial_level),  # Starting spot level.
            drift=float(spec.drift),                  # Drift parameter (mu).
            vol=float(spec.vol),                      # Volatility parameter (sigma).
            dt=float(spec.dt),                        # Time step size.
            initial_dispersion=float(spec.initial_dispersion),  # Dispersion at t=0 (optional).
        )

        # Store in quote panels with declared axes.
        state.quote_panels[market_id] = Panel(data=spot, axis_names=("time", "scenario"))
        # Cache raw spot array so FIXING can reuse exactly.
        state.spot_cache[market_id] = spot

    # ------------------------------------------------------------------
    # FIXING (reuse SPOT if available)
    # ------------------------------------------------------------------

    def generate_fixing(self, market_id: MarketId, state: SyntheticGenerationState) -> None:
        """
        Generate FX FIXING.

        Policy
        ------
        - Reuse the corresponding SPOT if already generated
        - Else, create a conservative constant time-series
        """
        # Build the corresponding SPOT MarketId for this FIXING MarketId.
        spot_id = MarketId(
            asset_class=market_id.asset_class,     # Same asset class ("FX").
            mkt_type="SPOT",                       # Spot type.
            name=market_id.name,                   # Same pair name (e.g., "EURUSD").
            qualifiers=market_id.qualifiers,       # Same qualifiers for consistent routing.
        )

        # If spot exists in cache, reuse it exactly.
        if spot_id in state.spot_cache:
            # Fetch the cached spot array [T,S].
            reused = state.spot_cache[spot_id]
            # Store it under the FIXING MarketId.
            state.quote_panels[market_id] = Panel(data=reused, axis_names=("time", "scenario"))
            return

        # If spot does not exist, fall back to a constant series [T,S].
        constant = np.full((int(state.n_time), int(state.n_scenarios)), 1.0, dtype=float)
        # Store constant fixing as a quote panel.
        state.quote_panels[market_id] = Panel(data=constant, axis_names=("time", "scenario"))

    # ------------------------------------------------------------------
    # VOL (forward-moneyness)
    # ------------------------------------------------------------------

    def generate_vol_grid_forward_moneyness(self, market_id: MarketId, state: SyntheticGenerationState) -> None:
        """
        Generate FX VOL surface on a grid [T,S,n_exp,n_k] in **absolute strike**,
        but *parameterized* by log-forward-moneyness:

            m(T,K) = log(K / F(T))

        where the forward is computed from spot and (domestic, foreign) curves:

            F(T) = S * exp((r_dom(T) - r_for(T)) * T)

        Storage (canonical)
        -------------------
        - vol_param_panels[mid] = Panel(data=[T,S,n_exp,n_k], axis_names=("time","scenario","expiry","strike"))
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

        # --- Resolve dependencies (spot + curves) ---
        # Build the corresponding SPOT MarketId (same name/qualifiers, mkt_type="SPOT").
        spot_id = MarketId(
            asset_class=market_id.asset_class,
            mkt_type="SPOT",
            name=market_id.name,
            qualifiers=market_id.qualifiers,
        )

        # Fetch the spot panel produced earlier in the engine (non-vol phase).
        spot_panel = state.quote_panels.get(spot_id)

        # Spot must exist; dependency closure + ordering should guarantee this.
        if spot_panel is None:
            raise ValueError(
                "FX VOL generation requires SPOT to be present (dependency closure should add it).\n"
                f"  missing spot_id={spot_id.key()}\n"
                f"  requested vol_id={market_id.key()}"
            )

        # Extract domestic/foreign currencies from qualifiers (if present).
        dom_ccy, for_ccy = _fx_dom_for_ccy(market_id)

        # If domestic currency exists, build the canonical curve id; else keep None.
        dom_curve_id = _default_ir_curve_id(dom_ccy) if dom_ccy is not None else None
        # If foreign currency exists, build the canonical curve id; else keep None.
        for_curve_id = _default_ir_curve_id(for_ccy) if for_ccy is not None else None

        # --- Generate the full cube ---
        vol_cube = _generate_fx_vol_cube_forward_moneyness(
            rng=rng,                                           # RNG for noise (and any stochasticity).
            spot_panel=np.asarray(spot_panel.data, dtype=float),  # Spot array [T,S].
            curve_params=state.curve_param_panels,             # Access to curve params panels.
            dom_curve_id=dom_curve_id,                         # Domestic curve id (or None).
            for_curve_id=for_curve_id,                         # Foreign curve id (or None).
            expiries=expiries,                                 # Expiry grid.
            strikes=strikes,                                   # Strike grid.
            atm_vol=float(spec.atm_vol),                       # Base ATM level.
            skew=float(spec.skew),                             # Linear moneyness term.
            smile=float(spec.smile),                           # Quadratic moneyness term.
            term=float(spec.term),                             # Term structure scaling.
            noise_scale=float(spec.noise_scale),               # Additive noise scale.
        )

        # Store the vol cube in canonical shape [T,S,n_exp,n_k] with semantic axis names.
        state.vol_param_panels[market_id] = Panel(
            data=vol_cube,
            axis_names=("time", "scenario", "expiry", "strike"),
        )

        # Store the factory needed to reconstruct a GridVolSurface at snapshot time.
        state.vol_factories[market_id] = GridVolFactory(
            expiries=expiries,                                 # Expiry grid used by factory.
            strikes=strikes,                                   # Strike grid used by factory.
            extrapolation=str(spec.extrapolation),             # Extrapolation policy.
        )


# -------------------------------------------------------------------------
# Helpers: qualifiers + curve lookup
# -------------------------------------------------------------------------

def _iter_qualifier_items(qualifiers) -> list[tuple[str, str]]:
    """
    Normalize MarketId.qualifiers into a list of (key, value) pairs.

    Why this exists
    ---------------
    Your codebase currently uses an immutable tuple-of-pairs representation:

        qualifiers=(("ccy","USD"), ("cut","NY"))

    However, some older / alternate code paths may pass qualifiers as a dict-like mapping.
    This helper makes FX generators desk-safe by accepting both without ambiguity.

    Accepted inputs
    ---------------
    - None
    - tuple/list of pairs: (("k","v"), ...)
    - Mapping: {"k": "v", ...}

    Returns
    -------
    list[(str,str)]
        A list of normalized (key, value) pairs, with whitespace stripped and keys lower-cased
        only at comparison time (we preserve the original strings here).
    """
    # If no qualifiers exist, return an empty list (common for simple MarketIds).
    if qualifiers is None:
        return []

    # Mapping-like (dict) case.
    if hasattr(qualifiers, "items"):
        out: list[tuple[str, str]] = []
        for k, v in qualifiers.items():  # type: ignore[attr-defined]
            out.append((str(k), str(v)))
        return out

    # Tuple-of-pairs (or list-of-pairs) case.
    out = []
    try:
        for kv in qualifiers:
            # Each element should be a 2-tuple (k, v).
            k, v = kv
            out.append((str(k), str(v)))
        return out
    except Exception as e:
        raise TypeError(
            "MarketId.qualifiers must be None, a mapping, or an iterable of (key,value) pairs. "
            f"Got type={type(qualifiers)!r}, value={qualifiers!r}."
        ) from e


def _fx_dom_for_ccy(market_id: MarketId) -> tuple[str | None, str | None]:
    """
    Extract domestic/foreign currencies from MarketId qualifiers.

    Accepted qualifier keys (case-insensitive)
    ----------------------------------------
    - Domestic: dom, domestic, dom_ccy
    - Foreign : for, foreign, for_ccy

    Returns
    -------
    (dom_ccy, for_ccy)
        Possibly (None, None) if not provided.

    Desk-grade note
    ---------------
    We do NOT infer dom/for from the FX pair string (EURUSD).
    We only read explicit qualifiers to avoid convention ambiguity across desks.
    """
    # Normalize qualifiers into a list of pairs regardless of representation.
    items = _iter_qualifier_items(market_id.qualifiers)

    def _get(*aliases: str) -> str | None:
        """
        Find the first matching qualifier key (by alias list), return its value.

        - Key match is case-insensitive.
        - Returned currency values are normalized to uppercase.
        """
        # Try each acceptable alias in priority order.
        for alias in aliases:
            alias_l = str(alias).strip().lower()

            # Scan all qualifier pairs.
            for k, v in items:
                if str(k).strip().lower() == alias_l:
                    ccy = str(v).strip().upper()
                    return ccy if ccy else None

        # Nothing matched any alias.
        return None

    dom = _get("dom", "domestic", "dom_ccy")
    foreign = _get("for", "foreign", "for_ccy")

    return dom, foreign


def _default_ir_curve_id(ccy: str) -> MarketId:
    """
    Construct a canonical IR curve MarketId for a currency.

    Canonical key format (matches your existing usage)
    -------------------------------------------------
      IR.CURVE.<CCY>.OIS|ccy=<CCY>

    Important implementation detail
    -------------------------------
    We return qualifiers as a tuple-of-pairs to match your canonical MarketId style and
    keep the object hashable/deterministic.

    Parameters
    ----------
    ccy:
        Currency code, e.g. "USD"

    Returns
    -------
    MarketId
        asset_class="IR", mkt_type="CURVE", name="<CCY>.OIS", qualifiers=(("ccy","<CCY>"),)
    """
    ccy_u = str(ccy).strip().upper()

    return MarketId(
        asset_class="IR",
        mkt_type="CURVE",
        name=f"{ccy_u}.OIS",
        qualifiers=(("ccy", ccy_u),),
    )


def _zero_rate_from_curve_params(
    *,
    curve_param_panels: dict[MarketId, Panel],
    curve_id: MarketId,
    time_idx: int,
    scenario_idx: int,
    expiry: float,
) -> float:
    """
    Read a continuous zero rate r(T) from stored curve params.

    Expected curve panel storage (canonical)
    ---------------------------------------
    params shaped: [T,S,K,2] with last dim columns [tenor, zero_rate].

    Behavior
    --------
    - If curve_id not present, caller should decide fallback.
    - We interpolate linearly in tenor with flat extrapolation.
    """
    # Get the curve param panel for this curve_id, if present.
    panel = curve_param_panels.get(curve_id)
    # If missing, raise KeyError (caller decides fallback policy).
    if panel is None:
        raise KeyError(curve_id.key())

    # Coerce underlying panel data to float ndarray.
    arr = np.asarray(panel.data, dtype=float)

    # Slice out the block for the requested (time, scenario): shape [K,2].
    block = arr[int(time_idx), int(scenario_idx), :, :]

    # Extract tenor grid (years) from first column.
    tenors = np.asarray(block[:, 0], dtype=float).reshape(-1)
    # Extract zero rates from second column.
    zeros = np.asarray(block[:, 1], dtype=float).reshape(-1)

    # Convert expiry to float for consistent comparisons.
    t = float(expiry)

    # Flat extrapolation on the left.
    if t <= float(tenors[0]):
        return float(zeros[0])

    # Flat extrapolation on the right.
    if t >= float(tenors[-1]):
        return float(zeros[-1])

    # Linear interpolation in tenor.
    return float(np.interp(t, tenors, zeros))


# -------------------------------------------------------------------------
# FX numeric generators
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
    Generate GBM spot panel [T,S]:

        S_{t+1} = S_t * exp((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)

    Notes
    -----
    - t=0 is deterministic across scenarios unless initial_dispersion > 0
    """
    # Validate input parameters for numerical stability.
    if initial_level <= 0.0:
        raise ValueError("initial_level must be > 0.")
    if vol < 0.0:
        raise ValueError("vol must be >= 0.")
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")

    # Convert sizes to ints for safety.
    t_count = int(n_time)
    s_count = int(n_scenarios)

    # Allocate output array [T,S].
    spot = np.empty((t_count, s_count), dtype=float)

    # Initial values across scenarios (optionally dispersed).
    if float(initial_dispersion) > 0.0:
        # Draw scenario-specific normals for initial dispersion.
        z0 = rng.normal(size=(s_count,))
        # Apply lognormal dispersion around initial_level.
        spot[0, :] = float(initial_level) * np.exp(float(initial_dispersion) * z0)
    else:
        # Same initial level for all scenarios.
        spot[0, :] = float(initial_level)

    # If only one time point, return immediately.
    if t_count == 1:
        return spot

    # Draw standard normals for each (time step, scenario).
    z = rng.normal(size=(t_count - 1, s_count))
    # Extract parameters as floats.
    mu = float(drift)
    sigma = float(vol)

    # Compute log-return increments for each step and scenario.
    log_returns = (mu - 0.5 * sigma * sigma) * float(dt) + sigma * np.sqrt(float(dt)) * z

    # Apply recurrence to build the full path.
    for t in range(1, t_count):
        spot[t, :] = spot[t - 1, :] * np.exp(log_returns[t - 1, :])

    return spot


def _generate_fx_vol_cube_forward_moneyness(
    *,
    rng: np.random.Generator,
    spot_panel: np.ndarray,
    curve_params: dict[MarketId, Panel],
    dom_curve_id: MarketId | None,
    for_curve_id: MarketId | None,
    expiries: np.ndarray,
    strikes: np.ndarray,
    atm_vol: float,
    skew: float,
    smile: float,
    term: float,
    noise_scale: float,
) -> np.ndarray:
    """
    Generate vol cube [T,S,n_exp,n_k] using log-forward-moneyness.

    Model (stable V1, market-aware)
    -------------------------------
    For each expiry T and strike K:
      F(T) = S * exp((r_dom(T) - r_for(T)) * T)
      m    = log(K / F(T))
      ATM(T) term structure: atm_vol * (1 + term*(sqrt(T) - sqrt(T_ref)))
      sigma(T,K) = ATM(T) * (1 + skew*m + smile*m^2)

    Curve fallback
    -------------
    If dom/for curves are missing, we treat r_dom=r_for=0 for that side.

    Noise
    -----
    Small additive Gaussian noise can be added per (time, scenario, expiry, strike).
    """
    # Ensure spot_panel is a float ndarray.
    spot = np.asarray(spot_panel, dtype=float)

    # Spot panel must be [T,S].
    if spot.ndim != 2:
        raise ValueError(f"spot_panel must be [T,S], got shape={spot.shape}.")

    # Extract (T,S) sizes from spot array.
    t_count, s_count = int(spot.shape[0]), int(spot.shape[1])

    # Normalize expiry/strike grids to 1D float arrays.
    exp = np.asarray(expiries, dtype=float).reshape(-1)
    k = np.asarray(strikes, dtype=float).reshape(-1)

    # Compute grid sizes.
    n_exp = int(exp.size)
    n_k = int(k.size)

    # Allocate output cube [T,S,n_exp,n_k].
    out = np.empty((t_count, s_count, n_exp, n_k), dtype=float)

    # Reference expiry for term scaling (avoid sqrt(0) instability).
    t_ref = float(exp[n_exp // 2])
    t_ref = max(t_ref, 1e-6)

    # Loop over all times.
    for ti in range(t_count):
        # Loop over all scenarios.
        for si in range(s_count):
            # Read spot for this (time, scenario).
            s0 = float(spot[ti, si])

            # Validate spot is finite and positive.
            if not np.isfinite(s0) or s0 <= 0.0:
                raise ValueError(f"Invalid spot encountered at (time,scenario)=({ti},{si}): S={s0}")

            # Loop over expiries.
            for ei in range(n_exp):
                # Expiry time in years.
                T = float(exp[ei])

                # --- Curve-based carry (continuous) ---
                # If domestic curve exists in storage, read r_dom(T); else use 0.
                if dom_curve_id is not None and dom_curve_id in curve_params:
                    r_dom = _zero_rate_from_curve_params(
                        curve_param_panels=curve_params,
                        curve_id=dom_curve_id,
                        time_idx=ti,
                        scenario_idx=si,
                        expiry=T,
                    )
                else:
                    r_dom = 0.0

                # If foreign curve exists in storage, read r_for(T); else use 0.
                if for_curve_id is not None and for_curve_id in curve_params:
                    r_for = _zero_rate_from_curve_params(
                        curve_param_panels=curve_params,
                        curve_id=for_curve_id,
                        time_idx=ti,
                        scenario_idx=si,
                        expiry=T,
                    )
                else:
                    r_for = 0.0

                # Forward for this expiry: F(T) = S * exp((r_dom - r_for) * T).
                F = float(s0 * np.exp((float(r_dom) - float(r_for)) * T))

                # Term structure on ATM vol (simple sqrt(T) scaling around T_ref).
                atm_T = float(atm_vol) * (1.0 + float(term) * (np.sqrt(max(T, 1e-6)) - np.sqrt(t_ref)))
                # Floor ATM to avoid degenerate vols.
                atm_T = max(atm_T, 1e-4)

                # Log-forward-moneyness m = log(K / F(T)) for all strikes.
                m = np.log(k / max(F, 1e-12))

                # Smile formula: sigma = ATM(T) * (1 + skew*m + smile*m^2).
                sigma = atm_T * (1.0 + float(skew) * m + float(smile) * (m ** 2))

                # Ensure positivity + floor.
                sigma = np.maximum(sigma, 1e-4)

                # Store the slice for this expiry across all strikes.
                out[ti, si, ei, :] = sigma

    # Add noise (vectorized) after deterministic structure.
    if float(noise_scale) > 0.0:
        # Draw Gaussian noise of same shape as out.
        out = out + rng.normal(loc=0.0, scale=float(noise_scale), size=out.shape)

    # Final safety floor to keep all vols positive.
    return np.maximum(out, 1e-4)