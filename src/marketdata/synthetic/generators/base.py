from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.marketdata.core.ids import MarketId
from src.marketdata.synthetic.context import SyntheticGenerationState
from src.marketdata.synthetic.schemas import MarketSchema


class Generator(Protocol):
    """
    Protocol for a synthetic generator.

    A generator:
      - declares the MarketSchema it supports (including dependencies)
      - writes outputs into the SyntheticRunContext containers

    Important
    ---------
    Generators should NOT return MarketDataset objects.
    They operate purely by writing Panels + factories into the run context.
    """

    def schema_for(self, mid: MarketId) -> MarketSchema:
        """Return the schema supported for the given MarketId (may depend on qualifiers)."""

    def generate(self, *, mid: MarketId, ctx: SyntheticGenerationState, seed: int) -> None:
        """Generate and store data for `mid` into the run context."""


@dataclass(frozen=True, slots=True)
class GeneratorResult:
    """
    Optional structured return for diagnostics (not required for storage).

    We keep this lightweight; the canonical outputs live in ctx.*_panels.
    """
    mid_key: str
    schema_id: str
    notes: str = ""