from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SyntheticMarketDataError(Exception):
    """
    Base exception for synthetic market data generation failures.

    Use this to distinguish "synthetic generation failed" from:
      - parsing errors
      - snapshot build errors
      - upstream provider errors
    """
    message: str

    def __str__(self) -> str:
        return str(self.message)


@dataclass(frozen=True, slots=True)
class MissingDependencyError(SyntheticMarketDataError):
    """Raised when a generator requires a dependency that was not produced."""


@dataclass(frozen=True, slots=True)
class UnknownMarketSchemaError(SyntheticMarketDataError):
    """Raised when no schema/generator mapping is registered for a MarketId."""


@dataclass(frozen=True, slots=True)
class InvalidPanelShapeError(SyntheticMarketDataError):
    """Raised when a generated panel does not match its declared schema."""