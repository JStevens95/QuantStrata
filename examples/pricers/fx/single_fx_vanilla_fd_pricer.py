from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption, AmericanFxVanillaOption

from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer
from src.pricers.fx.european_fde import FxEuropeanVanillaFdPricer
from src.pricers.fx.american_fde import FxAmericanVanillaFdPricer

from src.models.numeric.finite_difference.diagnostics import FdDiagnostics
from src.core.reporting.plots.pricers.finite_difference import (
    plot_price_curve_fd_vs_reference,
    plot_error_curve,
    plot_delta_profile,
    plot_gamma_profile,
    plot_surface_heatmap,
)

OptionType = Literal["call", "put"]


# ======================================================================================
# Minimal toy Market (matches your other examples/tests)
# ======================================================================================

@dataclass(frozen=True, slots=True)
class _FlatCurve:
    rate: float

    def df(self, t: float) -> float:
        t = float(t)
        if t < 0.0:
            raise ValueError("t must be >= 0.")
        return float(math.exp(-float(self.rate) * t))


@dataclass(frozen=True, slots=True)
class _FlatVolSurface:
    sigma: float

    def vol(self, *, expiry: float, strike: float) -> float:  # noqa: ARG002
        if float(expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")
        return float(self.sigma)


@dataclass(frozen=True, slots=True)
class _DummyMarket:
    spot: float
    rd: float
    rf: float
    sigma: float
    spot_id: MarketId
    vol_id: MarketId
    rd_id: MarketId
    rf_id: MarketId

    def quote(self, market_id: MarketId) -> float:
        if market_id != self.spot_id:
            raise KeyError(f"Unknown quote id: {market_id}")
        return float(self.spot)

    def curve(self, market_id: MarketId):
        if market_id == self.rd_id:
            return _FlatCurve(rate=self.rd)
        if market_id == self.rf_id:
            return _FlatCurve(rate=self.rf)
        raise KeyError(f"Unknown curve id: {market_id}")

    def vol_surface(self, market_id: MarketId):
        if market_id != self.vol_id:
            raise KeyError(f"Unknown vol id: {market_id}")
        return _FlatVolSurface(sigma=self.sigma)


# ======================================================================================
# Reference BSM curves per-unit on the FD spot grid
# ======================================================================================

def bsm_curves_per_unit(
    *,
    option_type: OptionType,
    spot_grid: np.ndarray,
    strike: float,
    expiry: float,
    rd: float,
    rf: float,
    sigma: float,
    ids: Dict[str, MarketId],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trade = EuropeanFxVanillaOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=1.0,
        strike=float(strike),
        expiry=float(expiry),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    bsm = FxEuropeanVanillaBsmPricer()

    pv = np.empty_like(spot_grid, dtype=np.float64)
    delta = np.empty_like(spot_grid, dtype=np.float64)
    gamma = np.empty_like(spot_grid, dtype=np.float64)

    for i, s in enumerate(np.asarray(spot_grid, dtype=np.float64)):
        m_i = _DummyMarket(
            spot=float(s),
            rd=float(rd),
            rf=float(rf),
            sigma=float(sigma),
            spot_id=ids["spot"],
            vol_id=ids["vol"],
            rd_id=ids["rd"],
            rf_id=ids["rf"],
        )
        pv[i] = float(bsm.price(trade, m_i))
        g = bsm.greeks(trade, m_i)
        delta[i] = float(g["delta"])
        gamma[i] = float(g["gamma"])

    return pv, delta, gamma


# ======================================================================================
# American-only diagnostics: boundary + intrinsic slices
# ======================================================================================

def extract_exercise_boundary(
    *,
    option_type: OptionType,
    spot_grid: np.ndarray,
    time_grid: np.ndarray,
    amer_surface: np.ndarray,
    strike: float,
    tol: float = 1e-8,
) -> np.ndarray:
    spot_grid = np.asarray(spot_grid, dtype=np.float64)
    time_grid = np.asarray(time_grid, dtype=np.float64)
    amer_surface = np.asarray(amer_surface, dtype=np.float64)

    intrinsic = (
        np.maximum(spot_grid - float(strike), 0.0)
        if option_type == "call"
        else np.maximum(float(strike) - spot_grid, 0.0)
    )

    s_star = np.full((time_grid.size,), np.nan, dtype=np.float64)

    for ti in range(time_grid.size):
        v = amer_surface[ti, :]
        active = intrinsic > 1e-14
        if not np.any(active):
            continue

        diff = v - intrinsic
        exercise = (diff <= float(tol)) & active
        if not np.any(exercise):
            continue

        idx = np.where(exercise)[0]
        if option_type == "put":
            s_star[ti] = float(spot_grid[idx.max()])
        else:
            s_star[ti] = float(spot_grid[idx.min()])

    return s_star


def plot_value_vs_intrinsic_slices(
    *,
    option_type: OptionType,
    spot_grid: np.ndarray,
    time_grid: np.ndarray,
    amer_surface: np.ndarray,
    strike: float,
    slice_times: Sequence[float],
    title_prefix: str,
) -> None:
    spot_grid = np.asarray(spot_grid, dtype=np.float64)
    time_grid = np.asarray(time_grid, dtype=np.float64)
    amer_surface = np.asarray(amer_surface, dtype=np.float64)

    intrinsic = (
        np.maximum(spot_grid - float(strike), 0.0)
        if option_type == "call"
        else np.maximum(float(strike) - spot_grid, 0.0)
    ).astype(np.float64, copy=False)

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(spot_grid, intrinsic, linestyle="--", label="Intrinsic")

    for t_sel in slice_times:
        ti = int(np.argmin(np.abs(time_grid - float(t_sel))))
        ax.plot(spot_grid, amer_surface[ti, :], label=f"V(t) at t≈{time_grid[ti]:.3f}")

    ax.set_title(f"{title_prefix} | American value vs intrinsic slices")
    ax.set_xlabel("Spot S")
    ax.set_ylabel("PV per unit notional")
    ax.legend()
    fig.tight_layout()


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    ids: Dict[str, MarketId] = {
        "spot": MarketId("FX", "SPOT", "EURUSD"),
        "vol": MarketId("FX", "VOL", "EURUSD.VOL"),
        "rd": MarketId("IR", "CURVE", "USD.OIS"),
        "rf": MarketId("IR", "CURVE", "EUR.OIS"),
    }

    # Pick params. For a clear exercise region: start with put.
    option_type: OptionType = "put"
    spot0 = 1.25
    strike = 1.25
    expiry = 1.0
    rd = 0.03
    rf = 0.01
    sigma = 0.20
    notional = 1_000_000.0

    market = _DummyMarket(
        spot=spot0,
        rd=rd,
        rf=rf,
        sigma=sigma,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )

    euro_trade = EuropeanFxVanillaOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=notional,
        strike=strike,
        expiry=expiry,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    amer_trade = AmericanFxVanillaOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=notional,
        strike=strike,
        expiry=expiry,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    # Pricers
    bsm = FxEuropeanVanillaBsmPricer()
    euro_fd = FxEuropeanVanillaFdPricer(n_space=401, n_time_steps=240, n_std=6.0, theta=0.5, use_log_space=True)
    amer_fd = FxAmericanVanillaFdPricer(n_space=401, n_time_steps=240, n_std=6.0, theta=0.5, use_log_space=True)

    # Scalar PVs (at S0)
    pv_bsm = float(bsm.price(euro_trade, market))
    pv_euro_fd = float(euro_fd.price(euro_trade, market))
    pv_amer_fd = float(amer_fd.price(amer_trade, market))

    print("=== PV ===")
    print(f"BSM (European): {pv_bsm:,.6f}")
    print(f"FD  (European): {pv_euro_fd:,.6f}")
    print(f"FD  (American) : {pv_amer_fd:,.6f}")
    print(f"Early exercise premium (Amer - Euro): {pv_amer_fd - pv_euro_fd:,.6f}")

    # Diagnostics (per unit, for plots)
    euro_diag: FdDiagnostics = euro_fd.diagnostics(euro_trade, market, store_surface=True)
    amer_diag: FdDiagnostics = amer_fd.diagnostics(amer_trade, market, store_surface=True)

    # BSM curves on the FD spot grid (per unit)
    pv_curve_bsm, delta_curve_bsm, gamma_curve_bsm = bsm_curves_per_unit(
        option_type=option_type,
        spot_grid=euro_diag.spot_grid,
        strike=strike,
        expiry=expiry,
        rd=rd,
        rf=rf,
        sigma=sigma,
        ids=ids,
    )

    title_prefix = (
        f"EURUSD {option_type.upper()} K={strike} T={expiry} | "
        f"n={euro_fd.n_space}x{euro_fd.n_time_steps} theta={euro_fd.theta}"
    )

    # =======================
    # EUROPEAN: FD vs BSM
    # =======================
    plot_price_curve_fd_vs_reference(
        out_path=None,
        spot_grid=euro_diag.spot_grid,
        fd_values_per_unit=euro_diag.values_t0_per_unit,
        ref_values_per_unit=pv_curve_bsm,
        spot0=spot0,
        title=f"{title_prefix} | European V(S,0): FD vs BSM (per unit)",
    )

    plot_error_curve(
        out_path=None,
        spot_grid=euro_diag.spot_grid,
        fd_values_per_unit=euro_diag.values_t0_per_unit,
        ref_values_per_unit=pv_curve_bsm,
        spot0=spot0,
        title=f"{title_prefix} | European error: FD - BSM (per unit)",
    )

    plot_delta_profile(
        out_path=None,
        spot_grid=euro_diag.spot_grid,
        fd_values_per_unit=euro_diag.values_t0_per_unit,
        ref_delta_per_unit=delta_curve_bsm,
        spot0=spot0,
        title=f"{title_prefix} | European delta(S): FD(surface-derivative) vs BSM",
    )

    plot_gamma_profile(
        out_path=None,
        spot_grid=euro_diag.spot_grid,
        fd_values_per_unit=euro_diag.values_t0_per_unit,
        ref_gamma_per_unit=gamma_curve_bsm,
        spot0=spot0,
        title=f"{title_prefix} | European gamma(S): FD(surface-derivative) vs BSM",
    )

    if euro_diag.surface_per_unit is not None:
        plot_surface_heatmap(
            out_path=None,
            spot_grid=euro_diag.spot_grid,
            time_grid=euro_diag.time_grid,
            surface=euro_diag.surface_per_unit,
            title=f"{title_prefix} | European FD surface V(t,S) (per unit)",
        )

    # =======================
    # AMERICAN: vs EURO FD
    # =======================
    plot_price_curve_fd_vs_reference(
        out_path=None,
        spot_grid=amer_diag.spot_grid,
        fd_values_per_unit=amer_diag.values_t0_per_unit,
        ref_values_per_unit=euro_diag.values_t0_per_unit,
        spot0=spot0,
        title=f"{title_prefix} | V(S,0): American FD vs European FD (per unit)",
    )

    plot_error_curve(
        out_path=None,
        spot_grid=amer_diag.spot_grid,
        fd_values_per_unit=amer_diag.values_t0_per_unit,
        ref_values_per_unit=euro_diag.values_t0_per_unit,
        spot0=spot0,
        title=f"{title_prefix} | Early exercise premium: American - European (per unit)",
    )

    if amer_diag.surface_per_unit is not None:
        plot_surface_heatmap(
            out_path=None,
            spot_grid=amer_diag.spot_grid,
            time_grid=amer_diag.time_grid,
            surface=amer_diag.surface_per_unit,
            title=f"{title_prefix} | American FD surface V(t,S) (per unit)",
        )

        s_star = extract_exercise_boundary(
            option_type=option_type,
            spot_grid=amer_diag.spot_grid,
            time_grid=amer_diag.time_grid,
            amer_surface=amer_diag.surface_per_unit,
            strike=strike,
            tol=1e-8,
        )

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(amer_diag.time_grid, s_star, marker="o", label="Estimated S*(t)")
        ax.set_title(f"{title_prefix} | Early exercise boundary S*(t)")
        ax.set_xlabel("Time t")
        ax.set_ylabel("Boundary spot S*")
        ax.legend()
        fig.tight_layout()

        plot_value_vs_intrinsic_slices(
            option_type=option_type,
            spot_grid=amer_diag.spot_grid,
            time_grid=amer_diag.time_grid,
            amer_surface=amer_diag.surface_per_unit,
            strike=strike,
            slice_times=[0.0, 0.25 * expiry, 0.5 * expiry, 0.75 * expiry],
            title_prefix=title_prefix,
        )

    plt.show()


if __name__ == "__main__":
    main()