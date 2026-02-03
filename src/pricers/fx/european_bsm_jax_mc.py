# src/pricers/fx/european_bsm_jax_mc.py
"""
JAX-based Monte Carlo pricer for European FX vanilla options (optional).

Uses JAX kernels for path/payoff when JAX is installed; enables CPU/GPU acceleration.
Register with pricer_id="jax_mc" when JAX is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.core.performance.backend import jax_available
from src.core.math.rates import rate_from_df as _rate_from_df
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.marketdata.core.market import Market
from src.models.payoffs.types import OptionType
from src.pricers.fx.european_bsm_mc import FxVanillaOptionMcSimulation


def _option_type_str(option_type: OptionType) -> str:
    """Map OptionType to 'call' | 'put' for JAX kernels."""
    if option_type in ("call", "put"):
        return str(option_type)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


@dataclass(frozen=True, slots=True)
class FxVanillaEuropeanJaxMcPricer:
    """
    Monte Carlo pricer for European FX vanilla options using JAX (optional backend).

    Same Garman–Kohlhagen mapping as FxVanillaEuropeanOptionMcPricer; uses JAX
    for path generation and payoff so it can run on GPU when available.

    Requires JAX installed. Use pricer_id="jax_mc" when resolving from the registry.
    """

    n_paths: int = 200_000
    seed: Optional[int] = 7
    antithetic: bool = True
    n_steps: int = 1
    scheme: str = "exact"

    def price(self, trade: FxVanillaEuropeanOption, market: Market) -> float:
        sim = self.run(trade, market, store_paths=False)
        return float(sim.discounted_payoffs.mean())

    def run(
        self,
        trade: FxVanillaEuropeanOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> FxVanillaOptionMcSimulation:
        if not jax_available():
            raise RuntimeError(
                "JAX is not installed. Use FxVanillaEuropeanOptionMcPricer or install: pip install jax jaxlib"
            )
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)

    def _run_simulation(
        self,
        trade: FxVanillaEuropeanOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> FxVanillaOptionMcSimulation:
        import jax
        from src.core.performance import jax_kernels

        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        option_type: OptionType = trade.option_type
        spot0 = float(market.quote(trade.spot_id))
        strike = float(trade.strike)
        maturity = float(trade.expiry)
        notional = float(trade.notional)

        if maturity < 0.0:
            raise ValueError("expiry must be >= 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")

        df_d = float(market.curve(trade.domestic_curve_id).df(maturity))
        df_f = float(market.curve(trade.foreign_curve_id).df(maturity))
        r_d = _rate_from_df(df=df_d, t=maturity)
        r_f = _rate_from_df(df=df_f, t=maturity)
        drift = float(r_d - r_f)
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        if maturity == 0.0:
            terminal_spots = np.array([spot0], dtype=np.float64)
            payoff_arr = np.maximum(terminal_spots - strike, 0.0) if option_type == "call" else np.maximum(strike - terminal_spots, 0.0)
            discounted_payoffs = (float(df_d) * payoff_arr * notional).astype(np.float64, copy=False)
            return FxVanillaOptionMcSimulation(
                spot0=spot0,
                strike=strike,
                maturity=maturity,
                df_domestic=df_d,
                drift=drift,
                sigma=sigma,
                option_type=option_type,
                notional=notional,
                n_paths_requested=self.n_paths,
                n_paths_effective=1,
                n_steps=0,
                scheme=self.scheme,
                antithetic=self.antithetic,
                seed=self.seed,
                terminal_spots=terminal_spots,
                discounted_payoffs=discounted_payoffs,
                paths=np.array([[spot0]], dtype=np.float64) if store_paths else None,
            )

        key = jax.random.PRNGKey(self.seed if self.seed is not None else 0)
        # JAX kernels do not implement antithetic; use n_paths (antithetic ignored for JAX)
        n_paths_eff = self.n_paths
        dt = maturity / self.n_steps

        terminal_jax = jax_kernels.gbm_terminal_spots_jax(
            spot0=spot0,
            drift=drift,
            vol=sigma,
            n_paths=n_paths_eff,
            n_steps=self.n_steps,
            dt=dt,
            key=key,
        )
        payoff_jax = jax_kernels.vanilla_payoff_jax(
            terminal_jax,
            strike=strike,
            option_type=_option_type_str(option_type),
        )
        discounted_jax = float(df_d) * payoff_jax * notional

        terminal_spots = np.asarray(terminal_jax, dtype=np.float64)
        discounted_payoffs = np.asarray(discounted_jax, dtype=np.float64)

        kept_paths = None
        if store_paths and paths_keep != 0:
            # JAX pricer does not store full paths by default; we only have terminal spots
            kept_paths = None

        return FxVanillaOptionMcSimulation(
            spot0=spot0,
            strike=strike,
            maturity=maturity,
            df_domestic=df_d,
            drift=drift,
            sigma=sigma,
            option_type=option_type,
            notional=notional,
            n_paths_requested=self.n_paths,
            n_paths_effective=n_paths_eff,
            n_steps=self.n_steps,
            scheme=self.scheme,
            antithetic=self.antithetic,
            seed=self.seed,
            terminal_spots=terminal_spots,
            discounted_payoffs=discounted_payoffs,
            paths=kept_paths,
        )
