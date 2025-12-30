from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Protocol, runtime_checkable

from src.marketdata.ids import MarketId


@runtime_checkable
class MarketView(Protocol):
    """
    Minimal interface required by pricers and scenario shocks.

    We use a Protocol so scenarios can return lightweight wrappers ("views")
    rather than mutating the concrete Market object.
    """

    def quote(self, market_id: MarketId) -> float:
        """Return a scalar quote (e.g., spot)."""
        ...

    def curve(self, market_id: MarketId):
        """Return a curve object (must provide df(t) at minimum)."""
        ...

    def vol_surface(self, market_id: MarketId):
        """Return a vol surface object (must provide vol(expiry, strike) at minimum)."""
        ...


class ScenarioShock(Protocol):
    """
    A scenario shock transforms a base MarketView into a new (shocked) MarketView.

    The key idea: we return a wrapper around the original market (a new view),
    rather than editing the base market in-place.
    """

    name: str

    def apply(self, base_market: MarketView) -> MarketView:
        """Return a shocked market view derived from the base market."""
        ...


@dataclass(frozen=True, slots=True)
class ScenarioPack:
    """
    Small convenience wrapper to apply multiple scenarios to one base market.

    This is helpful for orchestrators (price base + a set of shocked markets).
    """
    scenarios: Mapping[str, ScenarioShock]

    def apply_all(self, base_market: MarketView) -> Dict[str, MarketView]:
        """Apply each scenario to the base market and return a name -> shocked market map."""
        return {scenario_name: shock.apply(base_market) for scenario_name, shock in self.scenarios.items()}