"""
Exception hierarchy for the static replication library.

All exceptions inherit from StaticReplicationError so callers can
catch the full family with a single except clause when needed.
"""


class StaticReplicationError(Exception):
    """Base exception for the static replication library."""


class RegistryError(StaticReplicationError):
    """Raised when registry lookup or registration fails."""


class UnknownAssetClassError(StaticReplicationError):
    """Raised when no RiskFactorBuilder is registered for an asset class."""


class PipelineError(StaticReplicationError):
    """Raised when a pipeline stage fails during execution."""


class PnLComputationError(StaticReplicationError):
    """Raised when batch PnL computation fails for a factor group."""


class MarketDataError(StaticReplicationError):
    """Raised when market data loading or validation fails."""


class ShockError(StaticReplicationError):
    """Raised when shock/scenario loading or validation fails."""


class ScenarioError(StaticReplicationError):
    """Raised when scenario/shock generation fails."""


class PricingError(StaticReplicationError):
    """Raised when pricing kernel returns an invalid result."""


class TradeGenerationError(StaticReplicationError):
    """Raised when elementary trade generation fails for a cluster."""


class ValidationError(StaticReplicationError):
    """Raised when data validation checks fail (bounds, monotonicity, etc.)."""


class ConfigurationError(StaticReplicationError):
    """Raised when pipeline configuration is invalid or incomplete."""


class FactorResolutionError(StaticReplicationError):
    """Raised when risk-factor resolution fails for a trade or the portfolio."""
