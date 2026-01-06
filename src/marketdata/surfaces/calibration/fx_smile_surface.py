# src/marketdata/surfaces/calibration/fx_smile_to_grid.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np

from src.marketdata.surfaces.arbitrage import (
    FxSurfaceArbitrageConfig,
    check_fx_grid_surface_no_static_arb,
)
from src.marketdata.surfaces.quotes.fx_smile import FxSmileQuotes, FxSmileSliceQuotes
from src.marketdata.surfaces.vol_surface import GridVolSurface


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _forward_fx(*, spot: float, df_dom: float, df_for: float) -> float:
    """FX forward: F = S * df_for / df_dom."""
    s0 = float(spot)
    if s0 <= 0.0:
        raise ValueError("spot must be > 0.")
    dfd = float(df_dom)
    dff = float(df_for)
    if dfd <= 0.0 or dff <= 0.0:
        raise ValueError("discount factors must be > 0.")
    return float(s0 * dff / dfd)


def _interp_flat_logk(x_logk: np.ndarray, y: np.ndarray, xq_logk: np.ndarray) -> np.ndarray:
    """
    1D interpolation in log-strike with flat extrapolation at both ends.
    """
    x_logk = np.asarray(x_logk, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    xq_logk = np.asarray(xq_logk, dtype=float).reshape(-1)

    if x_logk.size == 0:
        raise ValueError("x_logk must be non-empty.")
    if x_logk.size != y.size:
        raise ValueError("x_logk and y must have the same length.")

    if x_logk.size == 1:
        return np.full_like(xq_logk, float(y[0]), dtype=float)

    # np.interp is linear; clamp queries for flat extrapolation
    xq = np.clip(xq_logk, x_logk[0], x_logk[-1])
    return np.interp(xq, x_logk, y).astype(float)


def _fixed_point_delta_strike(
    *,
    option_type: str,  # "call" | "put"
    abs_delta: float,
    expiry: float,
    spot: float,
    df_dom: float,
    df_for: float,
    surface: GridVolSurface,
    strike_from_abs_delta: Callable[..., float],
    vol0: float,
    max_iter: int,
    tol: float,
    damping: float,
) -> tuple[float, float]:
    """
    Solve for the strike K corresponding to an abs-delta quote when vol depends on K.

    We use a simple fixed-point iteration:
      K_n = K(delta, sigma_n)
      sigma_{n+1} = surface(T, K_n)

    Returns
    -------
    (K, sigma) at convergence (or last iteration).
    """
    t = float(expiry)
    if t <= 0.0:
        raise ValueError("expiry must be > 0 for delta-based inversion.")
    if not (0.0 < float(abs_delta) < 1.0):
        raise ValueError("abs_delta must be in (0,1).")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if not (0.0 <= damping <= 1.0):
        raise ValueError("damping must be in [0,1].")

    sigma = float(vol0)
    if sigma <= 0.0:
        sigma = float(surface.implied_vol(t, _forward_fx(spot=spot, df_dom=df_dom, df_for=df_for)))

    k = float("nan")
    for _ in range(int(max_iter)):
        k = float(
            strike_from_abs_delta(
                option_type=option_type,
                abs_delta=float(abs_delta),
                spot=float(spot),
                df_dom=float(df_dom),
                df_for=float(df_for),
                vol=float(sigma),
                expiry=float(t),
            )
        )
        sigma_new = float(surface.implied_vol(t, k))

        if abs(sigma_new - sigma) <= float(tol):
            return float(k), float(sigma_new)

        # damp update for stability (especially with steep skews)
        sigma = float((1.0 - damping) * sigma + damping * sigma_new)

    return float(k), float(sigma)


# -----------------------------------------------------------------------------
# Calibration: FX smile quotes -> GridVolSurface
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FxSmileToGridConfig:
    """
    Config for ATM/RR/BF FX smile -> (expiry, strike) grid surface.

    n_strikes:
      Number of strikes in the *base* log-moneyness grid (we also union in node strikes + forwards).
    moneyness_width:
      Base grid covers [S0*exp(-w), S0*exp(+w)] before unioning nodes.
    """
    n_strikes: int = 31
    moneyness_width: float = 0.35  # exp(±0.35) ~ [0.70, 1.42]
    extrapolation: str = "flat"     # GridVolSurface extrapolation


def calibrate_fx_smile_to_grid_surface(
    *,
    smile: FxSmileQuotes,
    spot: float,
    df_domestic: Callable[[float], float],
    df_foreign: Callable[[float], float],
    config: FxSmileToGridConfig = FxSmileToGridConfig(),
    surface_id: str | None = None,
    validate: bool = True,
    arb_config: FxSurfaceArbitrageConfig | None = None,
) -> GridVolSurface:
    """
    Convert FX smile quotes (ATM/RR/BF in delta space) into a GridVolSurface(T,K).
    """
    s0 = float(spot)
    if s0 <= 0.0:
        raise ValueError("spot must be > 0.")
    if config.n_strikes < 5:
        raise ValueError("config.n_strikes must be >= 5.")
    if config.moneyness_width <= 0.0:
        raise ValueError("config.moneyness_width must be > 0.")

    expiries = np.array(smile.expiries(), dtype=float)
    if np.any(expiries <= 0.0):
        raise ValueError("All expiries must be > 0 in V1 calibration.")

    # ---- Build per-expiry node sets ----
    nodes_by_expiry: List[Tuple[np.ndarray, np.ndarray]] = []  # (K_nodes, vol_nodes)
    all_k_nodes: List[float] = []
    all_fwds: List[float] = []

    for slc in smile:
        t = float(slc.expiry)
        df_d = float(df_domestic(t))
        df_f = float(df_foreign(t))
        fwd = _forward_fx(spot=s0, df_dom=df_d, df_for=df_f)
        all_fwds.append(float(fwd))

        # ATM node at forward
        k_list: List[float] = [float(fwd)]
        v_list: List[float] = [float(slc.atm_vol)]

        # delta wing nodes
        for d in slc.deltas():
            vol_put = float(slc.vol_put(d))
            vol_call = float(slc.vol_call(d))

            k_put = float(
                slc.delta_convention.strike_from_abs_delta(
                    option_type="put",
                    abs_delta=float(d),
                    spot=s0,
                    df_dom=df_d,
                    df_for=df_f,
                    vol=vol_put,
                    expiry=t,
                )
            )
            k_call = float(
                slc.delta_convention.strike_from_abs_delta(
                    option_type="call",
                    abs_delta=float(d),
                    spot=s0,
                    df_dom=df_d,
                    df_for=df_f,
                    vol=vol_call,
                    expiry=t,
                )
            )

            k_list.extend([k_put, k_call])
            v_list.extend([vol_put, vol_call])

        k_nodes = np.array(k_list, dtype=float)
        v_nodes = np.array(v_list, dtype=float)

        order = np.argsort(k_nodes)
        k_nodes = k_nodes[order]
        v_nodes = v_nodes[order]

        # de-dup nearly identical strikes (keep last)
        uniq_k: List[float] = []
        uniq_v: List[float] = []
        for kk, vv in zip(k_nodes.tolist(), v_nodes.tolist()):
            if not uniq_k or abs(kk - uniq_k[-1]) > 1e-12:
                uniq_k.append(float(kk))
                uniq_v.append(float(vv))
            else:
                uniq_v[-1] = float(vv)

        k_nodes = np.array(uniq_k, dtype=float)
        v_nodes = np.array(uniq_v, dtype=float)

        if np.any(~np.isfinite(v_nodes)) or np.any(v_nodes <= 0.0):
            raise ValueError("Non-positive or non-finite vol encountered in smile node conversion.")

        nodes_by_expiry.append((k_nodes, v_nodes))
        all_k_nodes.extend(k_nodes.tolist())

    # ---- Build common absolute strike grid ----
    w = float(config.moneyness_width)
    base = s0 * np.exp(np.linspace(-w, +w, int(config.n_strikes), dtype=float))

    strikes = np.unique(
        np.concatenate(
            [
                base.reshape(-1),
                np.array(all_k_nodes, dtype=float).reshape(-1),
                np.array(all_fwds, dtype=float).reshape(-1),
                np.array([s0], dtype=float),
            ]
        )
    )
    strikes = np.sort(strikes)
    if strikes.size < 5:
        raise ValueError("Strike grid ended up too small; check config / quotes.")

    # ---- Sample per-expiry ----
    vol_grid = np.empty((expiries.size, strikes.size), dtype=float)
    log_strikes = np.log(strikes)

    for i, (k_nodes, v_nodes) in enumerate(nodes_by_expiry):
        vol_grid[i, :] = _interp_flat_logk(np.log(k_nodes), v_nodes, log_strikes)
        vol_grid[i, :] = np.maximum(vol_grid[i, :], 1e-8)

    # ---- Arbitrage / sanity checks (optional) ----
    if validate:
        check_fx_grid_surface_no_static_arb(
            expiries=expiries,
            strikes=strikes,
            vols=vol_grid,
            spot=s0,
            df_domestic=df_domestic,
            df_foreign=df_foreign,
            config=arb_config or FxSurfaceArbitrageConfig(),
        )

    return GridVolSurface(
        expiries=expiries,
        strikes=strikes,
        implied_vols=vol_grid,
        extrapolation=str(config.extrapolation),
        strike_space="absolute",
        surface_id=surface_id,
    )


# -----------------------------------------------------------------------------
# Extraction: GridVolSurface -> FX smile quotes (ATM/RR/BF)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FxGridToSmileConfig:
    """
    Controls for extracting ATM/RR/BF quotes from a GridVolSurface.

    deltas:
        Which abs-deltas to extract (e.g. [0.25, 0.10]).
    max_iter / tol / damping:
        Fixed-point controls for delta->strike inversion under a strike-dependent vol surface.
    """
    deltas: Tuple[float, ...] = (0.25, 0.10)
    max_iter: int = 50
    tol: float = 1e-10
    damping: float = 0.5


def extract_fx_smile_from_grid_surface(
    *,
    surface: GridVolSurface,
    spot: float,
    df_domestic: Callable[[float], float],
    df_foreign: Callable[[float], float],
    config: FxGridToSmileConfig = FxGridToSmileConfig(),
    surface_id: str | None = None,
) -> FxSmileQuotes:
    """
    Extract ATM/RR/BF smile quotes for each expiry in a GridVolSurface.

    Output definition (market standard):
      RR(Δ) = σ_call(Δ) - σ_put(Δ)
      BF(Δ) = 0.5*(σ_call(Δ) + σ_put(Δ)) - σ_atm
    where σ_atm is taken at K=F(T) (ATM-forward convention in V1).
    """
    s0 = float(spot)
    if s0 <= 0.0:
        raise ValueError("spot must be > 0.")
    if surface.strike_space != "absolute":
        # You *can* relax this later, but for now keep it strict to avoid silent nonsense.
        raise ValueError(f"Expected surface.strike_space='absolute', got {surface.strike_space!r}.")

    deltas = [float(d) for d in config.deltas]
    if not deltas:
        raise ValueError("config.deltas must not be empty.")
    for d in deltas:
        if not (0.0 < d < 1.0):
            raise ValueError(f"All deltas must be in (0,1). Got {d}.")

    slices: List[FxSmileSliceQuotes] = []

    for t in surface.expiries.tolist():
        t = float(t)
        if t <= 0.0:
            # consistent with your calibration assumption (V1: expiries > 0)
            continue

        df_d = float(df_domestic(t))
        df_f = float(df_foreign(t))
        fwd = _forward_fx(spot=s0, df_dom=df_d, df_for=df_f)

        # ATM vol at forward strike
        atm_vol = float(surface.implied_vol(t, fwd))

        rr_by_delta: dict[float, float] = {}
        bf_by_delta: dict[float, float] = {}

        # Use a self-contained slice quote object to reuse its delta convention consistently.
        # (Your FxSmileSliceQuotes already carries delta_convention and atm_convention defaults.)
        tmp = FxSmileSliceQuotes(
            expiry=t,
            atm_vol=atm_vol,
            rr_by_delta={},
            bf_by_delta={},
            surface_id=surface_id,
        )

        for d in deltas:
            # Solve call wing
            _, sig_call = _fixed_point_delta_strike(
                option_type="call",
                abs_delta=d,
                expiry=t,
                spot=s0,
                df_dom=df_d,
                df_for=df_f,
                surface=surface,
                strike_from_abs_delta=tmp.delta_convention.strike_from_abs_delta,
                vol0=atm_vol,
                max_iter=config.max_iter,
                tol=config.tol,
                damping=config.damping,
            )

            # Solve put wing
            _, sig_put = _fixed_point_delta_strike(
                option_type="put",
                abs_delta=d,
                expiry=t,
                spot=s0,
                df_dom=df_d,
                df_for=df_f,
                surface=surface,
                strike_from_abs_delta=tmp.delta_convention.strike_from_abs_delta,
                vol0=atm_vol,
                max_iter=config.max_iter,
                tol=config.tol,
                damping=config.damping,
            )

            rr = float(sig_call - sig_put)
            bf = float(0.5 * (sig_call + sig_put) - atm_vol)

            rr_by_delta[float(d)] = rr
            bf_by_delta[float(d)] = bf

        slices.append(
            FxSmileSliceQuotes(
                expiry=t,
                atm_vol=atm_vol,
                rr_by_delta=rr_by_delta,
                bf_by_delta=bf_by_delta,
                delta_convention=tmp.delta_convention,
                atm_convention=tmp.atm_convention,
                surface_id=surface_id,
            )
        )

    return FxSmileQuotes(slices=slices)