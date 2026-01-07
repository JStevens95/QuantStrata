from __future__ import annotations

import math
from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.base import MarketView


# =============================================================================
# Market "views" (wrappers)
# =============================================================================
# These wrappers override ONE thing (quote / curve / vol surface) and delegate
# everything else to the underlying base market.
#
# This avoids any dependence on Market's internal storage structure and keeps
# shocks composable and safe.
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
class CurveWithParallelRateShift:
    """
    Curve wrapper that applies a parallel shift to continuous rates.

    If base df(t) = exp(-r t), then shifting the rate by +dr implies:
        df_shocked(t) = df_base(t) * exp(-dr * t)

    This works for ANY curve implementation as long as it provides df(t).
    """
    base_curve: object
    rate_shift: float

    def df(self, t: float) -> float:
        if t < 0.0:
            raise ValueError("df(t) requires t >= 0.")
        base_df = float(self.base_curve.df(t))
        return float(base_df * math.exp(-float(self.rate_shift) * float(t)))

    # Optional helper (nice for debugging / consistency with Curve Protocols).
    def zero_rate(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        df_t = float(self.df(t))
        return float(-math.log(df_t) / t)


@dataclass(frozen=True, slots=True)
class MarketWithOverriddenCurve:
    """A MarketView that overrides curve() for a single MarketId."""
    base_market: MarketView
    curve_id: MarketId
    overridden_curve: object

    def quote(self, market_id: MarketId) -> float:
        return float(self.base_market.quote(market_id))

    def curve(self, market_id: MarketId):
        if market_id == self.curve_id:
            return self.overridden_curve
        return self.base_market.curve(market_id)

    def vol_surface(self, market_id: MarketId):
        return self.base_market.vol_surface(market_id)


@dataclass(frozen=True, slots=True)
class VolSurfaceWithAdditiveBump:
    """
    VolSurface wrapper applying an additive bump to implied vol:
        sigma_shocked = max(sigma_base + bump, floor)

    Works for ANY surface implementing vol(expiry, strike).
    """
    base_surface: object
    vol_bump: float
    vol_floor: float = 1e-8

    def vol(self, expiry: float, strike: float) -> float:
        base_sigma = float(self.base_surface.vol(expiry, strike))
        bumped_sigma = base_sigma + float(self.vol_bump)
        return float(max(bumped_sigma, float(self.vol_floor)))


@dataclass(frozen=True, slots=True)
class VolSurfaceWithBump:
    """
    VolSurface wrapper applying either an absolute or relative bump to implied vol.

    - absolute: sigma' = max(sigma + bump, floor)
    - relative: sigma' = max(sigma * (1 + bump), floor)

    Works for ANY surface implementing vol(expiry, strike).
    """
    base_surface: object
    bump: float
    bump_mode: str = "relative"  # "relative" or "absolute"
    vol_floor: float = 1e-8

    def vol(self, expiry: float, strike: float) -> float:
        base_sigma = float(self.base_surface.vol(expiry, strike))

        if self.bump_mode == "relative":
            shocked_sigma = base_sigma * (1.0 + float(self.bump))
        elif self.bump_mode == "absolute":
            shocked_sigma = base_sigma + float(self.bump)
        else:
            raise ValueError("VolSurfaceWithBump.bump_mode must be 'relative' or 'absolute'.")

        return float(max(float(self.vol_floor), shocked_sigma))


@dataclass(frozen=True, slots=True)
class MarketWithOverriddenVolSurface:
    """A MarketView that overrides vol_surface() for a single MarketId."""
    base_market: MarketView
    vol_id: MarketId
    overridden_vol_surface: object

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
    bump_mode: str = "relative"  # "relative" or "absolute"

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
class FlatVolShock:
    """
    Shock an implied vol surface by an additive bump in sigma.

    Example: bump=+0.01 means 12% -> 13%.
    """
    name: str
    vol_id: MarketId
    vol_bump: float
    vol_floor: float = 1e-8

    def apply(self, base_market: MarketView) -> MarketView:
        base_surface = base_market.vol_surface(self.vol_id)

        shocked_surface = VolSurfaceWithAdditiveBump(
            base_surface=base_surface,
            vol_bump=float(self.vol_bump),
            vol_floor=float(self.vol_floor),
        )

        return MarketWithOverriddenVolSurface(
            base_market=base_market,
            vol_id=self.vol_id,
            overridden_vol_surface=shocked_surface,
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
    bump_mode: str = "relative"  # "relative" or "absolute"
    vol_floor: float = 1e-8

    def apply(self, base_market: MarketView) -> MarketView:
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