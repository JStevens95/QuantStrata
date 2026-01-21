from __future__ import annotations

# dataclass for concise class definitions.
from dataclasses import dataclass

# Typing for registry maps and call signatures.
from typing import Callable, Dict, Optional, Tuple

# MarketId is the routing key at runtime.
from src.marketdata.core.ids import MarketId

# SyntheticGenerationState is the shared mutable state passed to generators.
from src.marketdata.providers.synthetic.context import SyntheticGenerationState

# Generator signature used across the engine:
# a generator consumes (MarketId, state) and mutates state in-place (adds panels/factories/etc.).
GeneratorFn = Callable[[MarketId, SyntheticGenerationState], None]

# Requirements resolver returns prerequisite MarketIds for a requested MarketId.
RequirementsFn = Callable[[MarketId], Tuple[MarketId, ...]]


def _key(asset_class: str, mkt_type: str) -> Tuple[str, str]:
    """
    Normalize registry keys for deterministic lookup.

    We standardize on:
      (ASSET_CLASS_UPPER, MKT_TYPE_UPPER)
    so callers can register with any casing and still match at lookup time.
    """
    # Normalize asset_class and mkt_type by stripping whitespace and uppercasing.
    return (str(asset_class).strip().upper(), str(mkt_type).strip().upper())


@dataclass(slots=True)
class SyntheticRegistry:
    """
    Registry mapping (asset_class, mkt_type) -> generator, plus optional prerequisites.

    Why this exists
    ---------------
    - Keeps the SyntheticMarketEngine generic.
    - Lets you add asset classes by registering one module per asset class.
    - Enables desk-grade dependency closure (e.g., FX VOL requires FX SPOT + IR CURVES).

    Notes
    -----
    - We intentionally register by (asset_class, mkt_type) rather than MarketId.key()
      to keep the registry “schema-like” and avoid explosion of handlers.
    - The MarketId instance itself still contains pair/currency/cut/etc via name/qualifiers.
    """

    # Map from normalized (asset_class, mkt_type) to generation function.
    _generators: Dict[Tuple[str, str], GeneratorFn]
    # Map from normalized (asset_class, mkt_type) to dependency resolver.
    _requirements: Dict[Tuple[str, str], RequirementsFn]

    def __init__(self) -> None:
        # Initialize generator map as empty dict.
        self._generators = {}
        # Initialize requirement map as empty dict.
        self._requirements = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        asset_class: str,
        mkt_type: str,
        generator: GeneratorFn,
        requirements: Optional[RequirementsFn] = None,
    ) -> None:
        """
        Register a generator and optional prerequisite resolver.

        Parameters
        ----------
        requirements:
            If provided, must return MarketIds that *should be generated first*.
            The engine can use this to perform dependency closure.
        """
        # Build the normalized registry key.
        k = _key(asset_class, mkt_type)
        # Store the generator function for that key.
        self._generators[k] = generator

        # If requirements resolver provided, store it as well.
        if requirements is not None:
            self._requirements[k] = requirements

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, *, market_id: MarketId) -> Optional[GeneratorFn]:
        """
        Return the generator for this MarketId, or None if not registered.

        Lookup uses (market_id.asset_class, market_id.mkt_type) as the routing key.
        """
        # Normalize the MarketId fields and fetch the generator if present.
        return self._generators.get(_key(market_id.asset_class, market_id.mkt_type))

    def requirements(self, *, market_id: MarketId) -> Tuple[MarketId, ...]:
        """
        Return prerequisite MarketIds for this MarketId.

        If no resolver exists, returns empty tuple.
        """
        # Find the requirement resolver based on asset_class + mkt_type.
        fn = self._requirements.get(_key(market_id.asset_class, market_id.mkt_type))
        # If no resolver is registered, there are no prerequisites.
        if fn is None:
            return tuple()
        # Call resolver and ensure tuple output.
        reqs = fn(market_id)
        return tuple(reqs)