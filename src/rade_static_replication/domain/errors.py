"""
Exception hierarchy.

A shallow tree rooted at :class:`StaticReplicationError` so a caller can catch
everything this library raises with one ``except`` while still discriminating by
stage. Each stage raises the most specific subclass; the orchestrator tags
unexpected failures with the stage name.
"""
from __future__ import annotations


class StaticReplicationError(Exception):
    """Base class for every error raised by this library."""


class ConfigurationError(StaticReplicationError):
    """Malformed, missing, or inconsistent configuration."""


class ClientError(StaticReplicationError):
    """A portfolio/market-data client failed to return usable data."""


class PortfolioError(StaticReplicationError):
    """Raised during portfolio loading, normalisation, or validation."""


class FactorResolutionError(StaticReplicationError):
    """The risk-factor universe could not be resolved from the portfolio."""


class MarketDataError(StaticReplicationError):
    """A market-data object failed to build or failed its consistency checks."""


class BuilderError(StaticReplicationError):
    """An asset-class plugin is missing or failed while assembling factor data."""


class PricingError(StaticReplicationError):
    """A pricer or pricing kernel failed."""


class PnLError(StaticReplicationError):
    """The PnL engine failed to produce a scenario PnL matrix."""


class ClusteringError(StaticReplicationError):
    """Cluster resolution or artifact serialisation failed."""
