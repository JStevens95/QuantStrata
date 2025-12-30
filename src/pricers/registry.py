from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Type, TypeVar

# ---- Current instruments (you have now) ----
from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.options.european import EuropeanFxOption

# ---- Current pricers (you have now) ----
from src.pricers.analytic.black_scholes import BlackScholesPricer
from src.pricers.linear.spot import LinearFxSpotPricer
from src.pricers.linear.forward import LinearFxForwardPricer

T = TypeVar("T")


@dataclass(slots=True)
class DefaultPricerRegistry:
    """
    Default pricer registry for QuantStrata.

    Purpose
    -------
    Centralizes the mapping:
        Instrument class -> Pricer instance

    Why this is the right design (incremental complexity)
    -----------------------------------------------------
    - PortfolioPricer stays generic and never hardcodes model choice.
    - V1 starts minimal (FX spot + FX European option).
    - V2/V3 adds new instruments and pricers by registering new mappings
      without rewiring portfolio/stress/scenario infrastructure.

    Usage
    -----
    registry = DefaultPricerRegistry().build()
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)

    Extending
    ---------
    reg = DefaultPricerRegistry()
    reg.register(EqSpot, LinearEqSpotPricer())
    reg.register(EuropeanEqOption, BlackScholesPricer())
    registry = reg.build()
    """

    _registry: MutableMapping[Type[Any], Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Populate sensible V1 defaults. This keeps "works out of the box"
        # behavior while still allowing the user to override/extend.
        if not self._registry:
            self._registry.update(self._default_mappings())

    @staticmethod
    def _default_mappings() -> Dict[Type[Any], Any]:
        """
        Return the baseline instrument -> pricer mappings shipped with V1.

        Keep this minimal. Everything else can be added over time.
        """
        return {
            FxSpot: LinearFxSpotPricer(),
            FxForward: LinearFxForwardPricer(),
            EuropeanFxOption: BlackScholesPricer()
        }

    def register(self, instrument_type: Type[T], pricer: Any, *, overwrite: bool = False) -> None:
        """
        Register a pricer for an instrument class.

        Parameters
        ----------
        instrument_type:
            Class of the instrument (e.g., EuropeanFxOption).
        pricer:
            Pricer instance providing .price(...) and optionally .greeks(...).
        overwrite:
            If False and instrument_type already exists, raise ValueError.
        """
        if not isinstance(instrument_type, type):
            raise TypeError("instrument_type must be a class/type.")

        if instrument_type in self._registry and not overwrite:
            raise ValueError(
                f"Pricer already registered for {instrument_type.__name__}. "
                f"Pass overwrite=True to replace."
            )

        self._registry[instrument_type] = pricer

    def unregister(self, instrument_type: Type[Any]) -> None:
        """
        Remove a mapping if present (no-op if missing).
        """
        self._registry.pop(instrument_type, None)

    def build(self) -> Mapping[Type[Any], Any]:
        """
        Return an immutable view of the registry mapping.

        Notes
        -----
        PortfolioPricer only needs read access. Returning a Mapping discourages
        accidental mutation at runtime.
        """
        return dict(self._registry)

    def get(self, instrument_type: Type[Any]) -> Optional[Any]:
        """
        Convenience accessor (useful for debugging/tests).
        """
        return self._registry.get(instrument_type)