"""
Exception hierarchy for the static replication library.

All exceptions inherit from StaticReplicationError so callers can
catch the full family with a single except clause when needed.
"""


class StaticReplicationError(Exception):
    """Base exception for the static replication library."""


class UnknownAssetClassError(StaticReplicationError):
    """Raised when no RiskFactorBuilder is registered for an asset class."""


class PipelineError(StaticReplicationError):
    """Raised when a pipeline stage fails during execution."""


class PnLComputationError(StaticReplicationError):
    """Raised when batch PnL computation fails for a factor group."""


class MarketDataError(StaticReplicationError):
    """Raised when market data loading or validation fails."""


class ScenarioError(StaticReplicationError):
    """Raised when scenario/shock generation fails."""


class TradeGenerationError(StaticReplicationError):
    """Raised when elementary trade generation fails for a cluster."""


class ValidationError(StaticReplicationError):
    """Raised when data validation checks fail (bounds, monotonicity, etc.)."""


class ConfigurationError(StaticReplicationError):
    """Raised when pipeline configuration is invalid or incomplete."""
