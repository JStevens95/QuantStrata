"""
Base instrument abstractions.

All concrete instrument types (FX, IR, EQ, CR) inherit from
InstrumentSpec. This gives the trade generator, pricer, and
sensitivity calculator a uniform interface regardless of asset class.

Design:
  - to_pricer_params()  → flat dict for vectorised pricing
  - to_trade_id()       → deterministic globally-unique ID
  - price()             → single-instrument PV given market data
  - sensitivities()     → first-order greeks dict
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class InstrumentSpec(ABC):
    """Abstract base for all instrument specifications.

    Subclasses define the concrete fields for their asset class
    (e.g. strike, expiry, tenor, coupon) while inheriting the
    common interface that pricers, generators, and sensitivity
    calculators depend on.
    """
    asset_class: str
    payoff_type: str
    factor_id: str

    @abstractmethod
    def to_pricer_params(self) -> Dict[str, Any]:
        """Extract the parameter dict that the pricer needs.

        Returns a flat dict of numeric/string values that can be
        vectorised across a batch of trades.
        """
        ...

    @abstractmethod
    def to_trade_id(self) -> str:
        """Generate a deterministic, globally unique trade ID.

        Uses ``self.factor_id`` as the namespace since trades are
        generated at portfolio level before cluster assignment.
        """
        ...

    @abstractmethod
    def price(self, market_data: Any) -> float:
        """Compute present value of this instrument.

        Parameters
        ----------
        market_data : Any
            Asset-specific market data object (e.g. FXAsset, IRAsset)
            that provides spot, curves, vol surfaces needed for pricing.

        Returns
        -------
        float
            Present value (in domestic currency).
        """
        ...

    @abstractmethod
    def sensitivities(self, market_data: Any) -> Dict[str, float]:
        """Compute first-order sensitivities (greeks).

        Parameters
        ----------
        market_data : Any
            Same market data object as price().

        Returns
        -------
        dict
            Mapping of greek name → value.
            E.g. ``{"delta": 0.55, "gamma": 0.02, "vega": 0.12, "theta": -0.003}``
        """
        ...


@dataclass
class StrikeGrid:
    """User-defined grid of strikes for option generation.

    The trade generator resolves relative values (percentages of spot
    for FX, bp offsets from par rate for IR) into absolute strikes
    and stores them here.

    Parameters
    ----------
    values : list[float]
        Absolute strike values after resolution.
    convention : str
        How the strikes were specified: ``"absolute"``, ``"pct_of_spot"``,
        ``"bp_offset"``.
    reference_level : float
        The reference level strikes were resolved against (spot, par rate).
    """
    values: List[float] = field(default_factory=list)
    convention: str = "absolute"
    reference_level: float = 0.0


@dataclass
class TenorGrid:
    """User-defined grid of tenors/expiries/maturities for trade generation.

    Used for both FX option expiries and IR swap maturities / swaption
    expiries — they're all time axes in year fractions.

    Parameters
    ----------
    values : list[float]
        Tenor values in year fractions (e.g. ``[0.25, 0.5, 1.0, 2.0, 5.0]``).
    convention : str
        ``"year_fraction"`` (default) or ``"period_code"``.
    """
    values: List[float] = field(default_factory=list)
    convention: str = "year_fraction"
