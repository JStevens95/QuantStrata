from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple


# =============================================================================
# Helpers
# =============================================================================

def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _require_finite(name: str, x: float) -> float:
    x = float(x)
    if not _is_finite(x):
        raise ValueError(f"{name} must be finite. Got {x}.")
    return x


def _require_non_negative(name: str, x: float) -> float:
    x = _require_finite(name, x)
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0. Got {x}.")
    return x


def _require_positive(name: str, x: float) -> float:
    x = _require_finite(name, x)
    if x <= 0.0:
        raise ValueError(f"{name} must be > 0. Got {x}.")
    return x


def _sorted_unique(xs: Iterable[float]) -> List[float]:
    out = sorted({float(x) for x in xs})
    return out


DayCount = Literal["ACT/365F", "ACT/360", "30/360"]
FixedFreq = Literal["1Y", "6M", "3M", "1M"]


# =============================================================================
# Base protocol: "instrument quotes" are NOT curves
# =============================================================================

@dataclass(frozen=True, slots=True)
class RateQuote:
    """
    Base class for a single market rate quote used in curve construction.

    Design note
    -----------
    This is NOT a Curve. It does not define df(t). It is an *instrument quote*
    (deposit/FRA/future/swap/OIS/basis spread) at a specific maturity/interval.

    Builders/bootstrappers will take a list of these quotes and produce a Curve.
    """
    label: str  # human readable (e.g. "DEP 3M", "OIS 5Y", "FRA 3x6")

    def maturity(self) -> float:
        """Return a representative maturity time in years (for sorting/grouping)."""
        raise NotImplementedError


# =============================================================================
# Money-market deposits
# =============================================================================

@dataclass(frozen=True, slots=True)
class DepositQuote(RateQuote):
    """
    Money-market deposit quote.

    Parameters
    ----------
    t:
        Maturity in years (year fraction).
    rate:
        Simple annualized deposit rate (builder decides compounding convention).
    day_count:
        Day-count basis metadata (builder uses this if needed).
    """
    t: float
    rate: float
    day_count: DayCount = "ACT/365F"

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _require_positive("t", self.t))
        object.__setattr__(self, "rate", _require_finite("rate", self.rate))

    def maturity(self) -> float:
        return float(self.t)


# =============================================================================
# FRAs: forward rate agreements (T1 -> T2)
# =============================================================================

@dataclass(frozen=True, slots=True)
class FraQuote(RateQuote):
    """
    FRA quote on a forward period [t_start, t_end].

    Parameters
    ----------
    t_start, t_end:
        Year fractions, with 0 <= t_start < t_end.
    forward_rate:
        Annualized forward rate over the interval.
    day_count:
        Day-count basis metadata.
    """
    t_start: float
    t_end: float
    forward_rate: float
    day_count: DayCount = "ACT/365F"

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_start", _require_non_negative("t_start", self.t_start))
        object.__setattr__(self, "t_end", _require_positive("t_end", self.t_end))
        if float(self.t_end) <= float(self.t_start):
            raise ValueError(f"Require t_end > t_start. Got {self.t_start} -> {self.t_end}.")
        object.__setattr__(self, "forward_rate", _require_finite("forward_rate", self.forward_rate))

    def maturity(self) -> float:
        return float(self.t_end)


# =============================================================================
# Futures: typically quoted by price (e.g. 99.25 => implied rate 0.75%)
# =============================================================================

FuturesType = Literal["STIR"]  # keep V1 narrow; extend later (OIS futures etc.)

@dataclass(frozen=True, slots=True)
class FuturesQuote(RateQuote):
    """
    Short-rate (STIR) futures quote.

    V1 model
    --------
    - We store futures *price* and derive implied rate as (100 - price)/100.
    - Convexity adjustment is builder-specific (not done here).

    Parameters
    ----------
    t_start, t_end:
        Underlying accrual period (year fractions).
    price:
        Exchange-style price (e.g. 99.25).
    fut_type:
        Type tag for builder selection.
    """
    t_start: float
    t_end: float
    price: float
    fut_type: FuturesType = "STIR"

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_start", _require_non_negative("t_start", self.t_start))
        object.__setattr__(self, "t_end", _require_positive("t_end", self.t_end))
        if float(self.t_end) <= float(self.t_start):
            raise ValueError(f"Require t_end > t_start. Got {self.t_start} -> {self.t_end}.")
        object.__setattr__(self, "price", _require_finite("price", self.price))
        # Many STIR futures are around [90, 105]; don't over-restrict but guard nonsense.
        if not (0.0 < float(self.price) < 200.0):
            raise ValueError(f"Futures price looks invalid: {self.price}.")

    def implied_forward_rate(self) -> float:
        """Simple implied forward rate (no convexity adj): r = (100 - price) / 100."""
        return float((100.0 - float(self.price)) / 100.0)

    def maturity(self) -> float:
        return float(self.t_end)


# =============================================================================
# OIS and vanilla swaps: par rate quotes by maturity
# =============================================================================

SwapKind = Literal["OIS", "IRS"]


@dataclass(frozen=True, slots=True)
class ParSwapRateQuote(RateQuote):
    """
    Par swap rate quote.

    Notes
    -----
    - This is still a quote, not a curve.
    - The builder/bootstrapping code must know conventions (freq, day count, schedule, etc.).
    - 'index' is a tag used to route quotes to the correct bootstrapper.
    """
    kind: SwapKind
    maturity_t: float
    par_rate: float
    fixed_freq: FixedFreq = "1Y"
    fixed_day_count: DayCount = "30/360"
    index: Optional[str] = None  # e.g. "SOFR-OIS", "USD-LIBOR-3M" (for routing)

    def __post_init__(self) -> None:
        object.__setattr__(self, "maturity_t", _require_positive("maturity_t", self.maturity_t))
        object.__setattr__(self, "par_rate", _require_finite("par_rate", self.par_rate))

    def maturity(self) -> float:
        return float(self.maturity_t)


# =============================================================================
# Basis swap spreads (multi-curve): quote of spread between two float legs
# =============================================================================

@dataclass(frozen=True, slots=True)
class BasisSwapSpreadQuote(RateQuote):
    """
    Basis swap spread quote.

    Typical examples
    ----------------
    - 3M vs 6M basis
    - IBOR vs RFR basis (transition-era)
    - Cross-currency basis (would need CCY metadata; keep V1 single-ccy)

    Parameters
    ----------
    maturity_t:
        Maturity in years.
    spread:
        Quoted spread (typically added to one leg) as an annualized rate.
    leg_a, leg_b:
        Identifiers for the two floating indices (routing metadata).
    """
    maturity_t: float
    spread: float
    leg_a: str
    leg_b: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "maturity_t", _require_positive("maturity_t", self.maturity_t))
        object.__setattr__(self, "spread", _require_finite("spread", self.spread))
        if not self.leg_a or not self.leg_b:
            raise ValueError("leg_a and leg_b must be non-empty strings.")

    def maturity(self) -> float:
        return float(self.maturity_t)


# =============================================================================
# Containers: collections you actually pass around
# =============================================================================

@dataclass(frozen=True, slots=True)
class RateQuotes:
    """
    A typed container for multiple rate quotes used for curve construction.

    This is mainly for:
    - sorting / grouping
    - basic validation and convenience accessors
    - keeping examples readable

    Builders can accept:
      - RateQuotes
      - or just Sequence[RateQuote]
    """
    quotes: Tuple[RateQuote, ...]

    def __post_init__(self) -> None:
        if not self.quotes:
            raise ValueError("RateQuotes.quotes must not be empty.")

        # Basic sanity: maturities should be finite and non-negative
        ms = [q.maturity() for q in self.quotes]
        for m in ms:
            if not _is_finite(m) or m < 0.0:
                raise ValueError(f"Invalid quote maturity encountered: {m}")

    def maturities(self) -> List[float]:
        return _sorted_unique(q.maturity() for q in self.quotes)

    def sorted(self) -> "RateQuotes":
        return RateQuotes(quotes=tuple(sorted(self.quotes, key=lambda q: float(q.maturity()))))

    def filter_kind(self, kind: type) -> "RateQuotes":
        return RateQuotes(quotes=tuple(q for q in self.quotes if isinstance(q, kind)))

    def by_type(self) -> dict[str, List[RateQuote]]:
        out: dict[str, List[RateQuote]] = {}
        for q in self.quotes:
            out.setdefault(type(q).__name__, []).append(q)
        # sort each bucket by maturity for determinism
        for k in out:
            out[k] = sorted(out[k], key=lambda q: float(q.maturity()))
        return out


# =============================================================================
# Example factories (useful for docs/tests/examples)
# =============================================================================

def example_usd_ois_input_quotes() -> RateQuotes:
    """
    Example OIS bootstrapping inputs:
      - short deposits
      - OIS par rates out the curve

    These are illustrative numbers, not “market accurate”.
    """
    qs: List[RateQuote] = [
        DepositQuote(label="DEP 1W", t=7.0 / 365.0, rate=0.0480),
        DepositQuote(label="DEP 1M", t=1.0 / 12.0, rate=0.0485),
        DepositQuote(label="DEP 3M", t=0.25, rate=0.0490),
        ParSwapRateQuote(label="OIS 1Y", kind="OIS", maturity_t=1.0, par_rate=0.0488, index="SOFR-OIS"),
        ParSwapRateQuote(label="OIS 2Y", kind="OIS", maturity_t=2.0, par_rate=0.0465, index="SOFR-OIS"),
        ParSwapRateQuote(label="OIS 5Y", kind="OIS", maturity_t=5.0, par_rate=0.0430, index="SOFR-OIS"),
        ParSwapRateQuote(label="OIS 10Y", kind="OIS", maturity_t=10.0, par_rate=0.0410, index="SOFR-OIS"),
    ]
    return RateQuotes(quotes=tuple(qs)).sorted()


def example_usd_ibor_forward_inputs() -> RateQuotes:
    """
    Example multi-curve forward inputs:
      - FRAs and STIR futures as forward information

    In practice you’d also have basis quotes to connect discount vs forward curves.
    """
    qs: List[RateQuote] = [
        FraQuote(label="FRA 1x4", t_start=1.0 / 12.0, t_end=4.0 / 12.0, forward_rate=0.0510),
        FraQuote(label="FRA 3x6", t_start=0.25, t_end=0.50, forward_rate=0.0500),
        FuturesQuote(label="EDM4", t_start=0.75, t_end=1.00, price=95.25),
        FuturesQuote(label="EDU4", t_start=1.00, t_end=1.25, price=95.15),
    ]
    return RateQuotes(quotes=tuple(qs)).sorted()