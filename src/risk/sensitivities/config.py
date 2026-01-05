from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from src.marketdata.ids import MarketId


@dataclass(frozen=True, slots=True)
class SensitivitiesBumps:
    """
    Default bumps for FD sensitivities.

    Conventions
    -----------
    - spot_rel: relative bump to spot (e.g., 1e-4 = 1bp)
    - vol_abs : absolute bump to implied vol sigma (e.g., 1e-4)
    - rate_abs: absolute bump to continuous rate r (e.g., 1e-4 = 1bp)
    """
    spot_rel: float = 1e-4
    vol_abs: float = 1e-4
    rate_abs: float = 1e-4


@dataclass(frozen=True, slots=True)
class SensitivitiesConfig:
    """
    Configuration for the sensitivities engine.

    method
    ------
    - "analytic": read from PortfolioPricer.totals.greeks
    - "fd_central": bump-and-reprice using central differences

    Notes
    -----
    In V1 your pricers emit portfolio-level keys like:
      delta, gamma, vega, rho_domestic, rho_foreign

    For multi-underlying portfolios later, you’ll likely move to per-risk-factor
    keys (e.g. delta|EURUSD, rho|USD.OIS, vega|EURUSD), but this engine’s
    *output schema* stays stable either way.
    """
    method: str = "analytic"  # "analytic" or "fd_central"

    # IMPORTANT: avoid a shared mutable default instance
    bumps: SensitivitiesBumps = field(default_factory=SensitivitiesBumps)

    # Optional mapping for rate shocks -> greek key when using analytic method.
    # Example (FX): { rd_id: "rho_domestic", rf_id: "rho_foreign" }
    rho_key_by_curve_id: Optional[Mapping[MarketId, str]] = None

    # If analytic method is requested but the portfolio has multiple distinct
    # spot_ids/curve_ids/vol_ids and greeks are not keyed per-id, fall back to FD.
    fallback_to_fd_when_ambiguous: bool = True