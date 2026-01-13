from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Sequence

from src.marketdata.core.panel import Panel


def make_time_grid(
    *,
    start: str | np.datetime64,
    n_t: int,
    step: str = "D",
) -> list[str]:
    """
    Create a deterministic time grid (list[str]) for MarketDataset.dates.

    Parameters
    ----------
    start:
        Start date (e.g. "2026-01-07") or np.datetime64.
    n_t:
        Number of time points (T).
    step:
        Numpy timedelta unit, e.g. "D" for days.

    Returns
    -------
    list[str]
        ISO date strings suitable for MarketDataset.dates.
    """
    if int(n_t) <= 0:
        raise ValueError("make_time_grid: n_t must be >= 1.")

    base = np.datetime64(start)
    dates = [(base + np.timedelta64(i, step)).astype(str) for i in range(int(n_t))]
    return dates


def make_quote_panel(
    *,
    n_t: int,
    n_s: int,
    values: np.ndarray | Sequence[float] | Callable[[int, int], float],
    axis_names: tuple[str, ...] | None = None,
    include_scenario_axis_when_s1: bool = True,
) -> Panel:
    """
    Build a quote Panel, intended for scalar Market.quote() retrieval.

    Supports:
      - callable values(ti, si) -> float, builds [T,S]
      - array-like:
          [T]        -> if n_s==1 and include_scenario_axis_when_s1==False
          [T,S]      -> scenario-aware

    Notes
    -----
    - If output is 2D, axis_names MUST be ("time","scenario") to satisfy dataset invariants.
    """
    if int(n_t) <= 0:
        raise ValueError("make_quote_panel: n_t must be >= 1.")
    if int(n_s) <= 0:
        raise ValueError("make_quote_panel: n_s must be >= 1.")

    T = int(n_t)
    S = int(n_s)

    if callable(values):
        data = np.empty((T, S), dtype=float)
        for ti in range(T):
            for si in range(S):
                data[ti, si] = float(values(ti, si))
        ax = axis_names or ("time", "scenario")
        return Panel(data=np.asarray(data, dtype=float), axis_names=ax)

    arr = np.asarray(values, dtype=float)

    if arr.ndim == 1:
        if arr.shape[0] != T:
            raise ValueError(f"make_quote_panel: expected shape [T]={T}, got {arr.shape}.")
        if S == 1 and not include_scenario_axis_when_s1:
            ax = axis_names or ("time",)
            return Panel(data=arr, axis_names=ax)
        # promote to [T,1]
        data2 = arr.reshape(T, 1)
        ax = axis_names or ("time", "scenario")
        return Panel(data=data2, axis_names=ax)

    if arr.ndim == 2:
        if arr.shape != (T, S):
            raise ValueError(f"make_quote_panel: expected shape [T,S]=({T},{S}), got {arr.shape}.")
        ax = axis_names or ("time", "scenario")
        return Panel(data=arr, axis_names=ax)

    raise ValueError(f"make_quote_panel: values must be callable, [T] or [T,S]; got ndim={arr.ndim}.")


def make_zero_curve_panel(
    *,
    dates: Sequence[str],
    n_s: int,
    tenors: np.ndarray | Sequence[float],
    zero_rates: np.ndarray | Sequence[float] | Callable[[int, int], np.ndarray],
    axis_names: tuple[str, ...] = ("time", "scenario", "tenor", "field"),
) -> Panel:
    """
    Build a curve parameter Panel compatible with ZeroRateCurveFactory.

    Output shape:
      [T, S, K, 2]
        [:,:,:,0] = tenor
        [:,:,:,1] = zero_rate

    This is intentionally "factory-friendly": MarketDataset.snapshot slices to [K,2].
    """
    dates = list(dates)
    T = len(dates)
    if T == 0:
        raise ValueError("make_zero_curve_panel: dates must be non-empty.")
    if int(n_s) <= 0:
        raise ValueError("make_zero_curve_panel: n_s must be >= 1.")
    S = int(n_s)

    ten = np.asarray(tenors, dtype=float).reshape(-1)
    if ten.size == 0:
        raise ValueError("make_zero_curve_panel: tenors must be non-empty.")
    K = int(ten.size)

    out = np.empty((T, S, K, 2), dtype=float)
    out[:, :, :, 0] = ten.reshape(1, 1, K)

    if callable(zero_rates):
        for ti in range(T):
            for si in range(S):
                z = np.asarray(zero_rates(ti, si), dtype=float).reshape(-1)
                if z.size != K:
                    raise ValueError(
                        f"make_zero_curve_panel: zero_rates(ti,si) must have size K={K}; got {z.size}."
                    )
                out[ti, si, :, 1] = z
    else:
        base = np.asarray(zero_rates, dtype=float).reshape(-1)
        if base.size != K:
            raise ValueError(f"make_zero_curve_panel: zero_rates must have size K={K}; got {base.size}.")
        out[:, :, :, 1] = base.reshape(1, 1, K)

    return Panel(data=out, axis_names=axis_names)


def make_grid_vol_panel(
    *,
    dates: Sequence[str],
    n_s: int,
    spot_panel: Panel,
    expiries: np.ndarray | Sequence[float],
    strikes: np.ndarray | Sequence[float],
    vol_grid: Optional[Callable[[float, int, int, np.ndarray, np.ndarray], np.ndarray]] = None,
    axis_names: tuple[str, ...] = ("time", "scenario", "expiry", "strike"),
    vol_floor: float = 1e-8,
) -> Panel:
    """
    Build a grid-vol parameter Panel.

    Output shape:
      [T, S, n_exp, n_k]

    Parameters
    ----------
    spot_panel:
        Quote Panel used to supply spot(t,s) for smile construction.
        Must be [T,S] or [T] (in which case S must be 1).
    vol_grid:
        Optional callable to generate a full [n_exp, n_k] grid:
            vol_grid(spot, ti, si, expiries, strikes) -> np.ndarray[n_exp, n_k]
        If None, uses a simple synthetic skew/curvature demo smile.

    Notes
    -----
    This panel is consumed by GridVolFactory which accepts either:
      - shape (n_exp, n_k) block
      - or flattened n_exp*n_k
    """
    dates = list(dates)
    T = len(dates)
    if T == 0:
        raise ValueError("make_grid_vol_panel: dates must be non-empty.")
    if int(n_s) <= 0:
        raise ValueError("make_grid_vol_panel: n_s must be >= 1.")
    S = int(n_s)

    exp = np.asarray(expiries, dtype=float).reshape(-1)
    k = np.asarray(strikes, dtype=float).reshape(-1)
    if exp.size == 0 or k.size == 0:
        raise ValueError("make_grid_vol_panel: expiries/strikes must be non-empty.")

    n_exp = int(exp.size)
    n_k = int(k.size)

    # Extract spot time series in a consistent [T,S] form
    spot_arr = np.asarray(spot_panel.data, dtype=float)
    if spot_arr.ndim == 1:
        if S != 1 or spot_arr.shape[0] != T:
            raise ValueError("make_grid_vol_panel: spot_panel [T] requires n_s=1 and matching T.")
        spot_ts = spot_arr.reshape(T, 1)
    elif spot_arr.ndim == 2:
        if spot_arr.shape != (T, S):
            raise ValueError(
                f"make_grid_vol_panel: spot_panel shape must be (T,S)=({T},{S}); got {spot_arr.shape}."
            )
        spot_ts = spot_arr
    else:
        raise ValueError("make_grid_vol_panel: spot_panel must be [T] or [T,S].")

    # Default synthetic smile if no generator provided.
    def _default_smile(spot: float, ti: int, si: int, expiries_: np.ndarray, strikes_: np.ndarray) -> np.ndarray:
        log_m = np.log(strikes_ / float(spot))
        out_ = np.empty((expiries_.size, strikes_.size), dtype=float)

        # very mild time/scenario drift in ATM just to make panels non-flat
        atm_base = 0.12 + 0.002 * (ti / max(T - 1, 1)) + 0.002 * si

        for i, Texp in enumerate(expiries_.tolist()):
            skew = -0.06 * np.sqrt(float(Texp))
            curv = 0.18 / (1.0 + float(Texp))
            out_[i, :] = float(atm_base) + skew * log_m + curv * (log_m ** 2)

        return out_

    gen = vol_grid or _default_smile

    out = np.empty((T, S, n_exp, n_k), dtype=float)
    for ti in range(T):
        for si in range(S):
            spot = float(spot_ts[ti, si])
            grid = np.asarray(gen(spot, ti, si, exp, k), dtype=float)
            if grid.shape != (n_exp, n_k):
                raise ValueError(
                    f"make_grid_vol_panel: vol_grid returned shape {grid.shape}, expected {(n_exp, n_k)}."
                )
            out[ti, si, :, :] = np.maximum(grid, float(vol_floor))

    return Panel(data=out, axis_names=axis_names)