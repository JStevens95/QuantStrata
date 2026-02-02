from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Curve, VolSurface
from src.marketdata.scenarios.interfaces import MarketView, ScenarioShock

BumpMode = Literal["relative", "absolute"]


# =============================================================================
# Market "views" (wrappers)
# =============================================================================

@dataclass(frozen=True, slots=True)
class MarketWithOverriddenQuote:
    """A MarketView that overrides quote() for a single MarketId."""
    base_market: MarketView
    quote_id: MarketId
    overridden_quote_value: float

    def quote(self, market_id: MarketId) -> float:
        if market_id == self.quote_id:
            return float(self.overridden_quote_value)
        return float(self.base_market.quote(market_id))

    def curve(self, market_id: MarketId):
        return self.base_market.curve(market_id)

    def vol_surface(self, market_id: MarketId):
        return self.base_market.vol_surface(market_id)


@dataclass(frozen=True, slots=True)
class CurveWithParallelRateShift(Curve):
    """
    Curve wrapper that applies a parallel shift to continuous rates.

    If base df(t) = exp(-r t), then shifting the rate by +dr implies:
        df_shocked(t) = df_base(t) * exp(-dr * t)
    """
    base_curve: Curve
    rate_shift: float

    def df(self, t: float) -> float:
        t = float(t)
        if t < 0.0:
            raise ValueError("df(t) requires t >= 0.")
        base_df = float(self.base_curve.df(t))
        return float(base_df * math.exp(-float(self.rate_shift) * t))

    def zero_rate(self, t: float) -> float:
        t = float(t)
        if t <= 0.0:
            return 0.0
        df_t = float(self.df(t))
        return float(-math.log(df_t) / t)

    def forward_rate(self, t1: float, t2: float) -> float:
        t1 = float(t1)
        t2 = float(t2)
        if t1 < 0.0 or t2 < 0.0:
            raise ValueError("forward_rate(t1,t2) requires t1,t2 >= 0.")
        if t2 <= t1:
            raise ValueError("forward_rate(t1,t2) requires t2 > t1.")
        return float(math.log(self.df(t1) / self.df(t2)) / (t2 - t1))

    def __getattr__(self, name: str):
        # Delegate any curve-specific helpers to the base curve.
        return getattr(self.base_curve, name)


@dataclass(frozen=True, slots=True)
class MarketWithOverriddenCurve:
    """A MarketView that overrides curve() for a single MarketId."""
    base_market: MarketView
    curve_id: MarketId
    overridden_curve: Curve

    def quote(self, market_id: MarketId) -> float:
        return float(self.base_market.quote(market_id))

    def curve(self, market_id: MarketId):
        if market_id == self.curve_id:
            return self.overridden_curve
        return self.base_market.curve(market_id)

    def vol_surface(self, market_id: MarketId):
        return self.base_market.vol_surface(market_id)


@dataclass(frozen=True, slots=True)
class VolSurfaceWithBump(VolSurface):
    """
    VolSurface wrapper applying either an absolute or relative bump to implied vol.

    - absolute: sigma' = max(sigma + bump, floor)
    - relative: sigma' = max(sigma * (1 + bump), floor)
    """
    base_surface: VolSurface
    bump: float
    bump_mode: BumpMode = "relative"
    vol_floor: float = 1e-8

    def implied_vol(self, expiry: float, strike: float) -> float:
        base_sigma = float(self.base_surface.implied_vol(float(expiry), float(strike)))

        if self.bump_mode == "relative":
            shocked_sigma = base_sigma * (1.0 + float(self.bump))
        elif self.bump_mode == "absolute":
            shocked_sigma = base_sigma + float(self.bump)
        else:
            raise ValueError("VolSurfaceWithBump.bump_mode must be 'relative' or 'absolute'.")

        return float(max(float(self.vol_floor), shocked_sigma))

    def vol(self, expiry: float, strike: float) -> float:
        # Compatibility alias required by your VolSurface Protocol
        return float(self.implied_vol(expiry, strike))

    def __getattr__(self, name: str):
        # Delegate any surface-specific helpers/metadata.
        return getattr(self.base_surface, name)


@dataclass(frozen=True, slots=True)
class MarketWithOverriddenVolSurface:
    """A MarketView that overrides vol_surface() for a single MarketId."""
    base_market: MarketView
    vol_id: MarketId
    overridden_vol_surface: VolSurface

    def quote(self, market_id: MarketId) -> float:
        return float(self.base_market.quote(market_id))

    def curve(self, market_id: MarketId):
        return self.base_market.curve(market_id)

    def vol_surface(self, market_id: MarketId):
        if market_id == self.vol_id:
            return self.overridden_vol_surface
        return self.base_market.vol_surface(market_id)


# =============================================================================
# Scenario shocks
# =============================================================================

@dataclass(frozen=True, slots=True)
class SpotShock:
    """
    Shock a spot (or any scalar quote) by either a relative or absolute bump.

    - relative: shocked = base * (1 + bump)
    - absolute: shocked = base + bump
    """
    name: str
    spot_id: MarketId
    bump: float
    bump_mode: BumpMode = "relative"

    def apply(self, base_market: MarketView) -> MarketView:
        base_spot = float(base_market.quote(self.spot_id))

        if self.bump_mode == "relative":
            shocked_spot = base_spot * (1.0 + float(self.bump))
        elif self.bump_mode == "absolute":
            shocked_spot = base_spot + float(self.bump)
        else:
            raise ValueError("SpotShock.bump_mode must be 'relative' or 'absolute'.")

        return MarketWithOverriddenQuote(
            base_market=base_market,
            quote_id=self.spot_id,
            overridden_quote_value=shocked_spot,
        )


@dataclass(frozen=True, slots=True)
class VolShock:
    """
    Shock an implied vol surface by either a relative or absolute bump.

    Examples
    --------
    - bump_mode="absolute", bump=+0.01: 12% -> 13%
    - bump_mode="relative", bump=+0.10: 12% -> 13.2%
    """
    name: str
    vol_id: MarketId
    bump: float
    bump_mode: BumpMode = "relative"
    vol_floor: float = 1e-8

    def apply(self, base_market: MarketView) -> MarketView:
        if self.bump_mode not in ("relative", "absolute"):
            raise ValueError("VolShock.bump_mode must be 'relative' or 'absolute'.")

        base_surface = base_market.vol_surface(self.vol_id)

        shocked_surface = VolSurfaceWithBump(
            base_surface=base_surface,
            bump=float(self.bump),
            bump_mode=str(self.bump_mode),
            vol_floor=float(self.vol_floor),
        )

        return MarketWithOverriddenVolSurface(
            base_market=base_market,
            vol_id=self.vol_id,
            overridden_vol_surface=shocked_surface,
        )


@dataclass(frozen=True, slots=True)
class ParallelRateShock:
    """
    Parallel shift a discount curve in continuous rates.

    Example: rate_shift=+0.01 means +100bp shift.
    """
    name: str
    curve_id: MarketId
    rate_shift: float

    def apply(self, base_market: MarketView) -> MarketView:
        base_curve = base_market.curve(self.curve_id)

        shocked_curve = CurveWithParallelRateShift(
            base_curve=base_curve,
            rate_shift=float(self.rate_shift),
        )

        return MarketWithOverriddenCurve(
            base_market=base_market,
            curve_id=self.curve_id,
            overridden_curve=shocked_curve,
        )


@dataclass(frozen=True, slots=True)
class CompositeShock:
    """
    Multi-factor scenario: apply a sequence of shocks in order.

    Each shock is applied to the result of the previous one, so the final
    market view has all shocks applied. Implements ScenarioShock protocol.
    """

    name: str
    shocks: Sequence[ScenarioShock]

    def apply(self, base_market: MarketView) -> MarketView:
        current = base_market
        for shock in self.shocks:
            current = shock.apply(current)
        return current