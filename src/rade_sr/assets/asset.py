"""
Base asset class — self-contained market data + shocks for one risk factor.

After load(), an Asset holds everything the pipeline needs:
  - Spot price + historical time series  (universal)
  - Typed shock object                   (universal)
  - Dependent assets                     (universal, may be empty)
  - Asset-class-specific market data     (subclass-owned)

What lives WHERE:
  Base (Asset)        Subclass
  ──────────────      ────────────────────────────────
  spot                FXAsset.vol_surface, .forward_points
  spot_series         IRAsset.curve, .vol_cube, .rate_history
  shocks (typed)      EQAsset.vol_surface, .dividends
  dependencies        CRAsset.credit_spread_curve

Loading sequence (three hooks):
  1. _load_dependencies()  — other Assets this one needs (called first)
  2. _load()               — ALL market data: spot, time series, vol,
                             curves, forward points — everything.
                             Subclass owns the order.
  3. _load_shocks()        — shock scenarios (separate because shocks
                             need market data grids to exist for axis
                             alignment)

Design principles:
  - Asset does NOT manage file paths or I/O directly.
  - Immutable config separated from mutable market state.
  - Dependencies are full Asset instances.
  - Shocks are typed dataclasses with labelled axes.
  - Vol shape is asset-class-specific.  The base class does NOT
    impose a vol field.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from .types import AssetConfig

logger = logging.getLogger(__name__)


class Asset(ABC):
    """Abstract base: self-contained market data + shocks for one risk factor.

    Lifecycle:
        1. Construct with AssetConfig (immutable identity).
        2. Call load(api_client) — runs the three hooks in order.
        3. Pipeline reads .shocks, dependent assets, market data.

    Subclasses implement three hooks + validate:
        _load_dependencies()  — other Assets this one needs
        _load()               — all market data (spot, vol, curves, etc.)
        _load_shocks()        — typed shock dataclass

    After load() completes, at minimum these must be set:
        self.spot           — reference price / rate
        self.spot_series    — historical time series
        self._shocks        — typed shock object (FXShocks, IRShocks, etc.)
    """

    def __init__(self, config: AssetConfig) -> None:
        self.config = config

        # Universal fields (populated by load hooks)
        self.spot: Optional[float] = None
        self.spot_series: Optional[pd.Series] = None
        self._shocks: Any = None
        self._dependencies: Dict[str, Asset] = {}
        self._loaded = False

    # ── Public interface ──────────────────────────────────────────────

    @property
    def asset_class(self) -> str:
        return self.config.asset_class

    @property
    def asset_name(self) -> str:
        return self.config.asset_name

    @property
    def factor_id(self) -> str:
        """Canonical ID for pipeline registry lookups."""
        return self.config.asset_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def shocks(self) -> Any:
        """Typed shock object (FXShocks, IRShocks, etc.).

        Subclasses narrow the return type.
        """
        return self._shocks

    @property
    def dependent_assets(self) -> Dict[str, "Asset"]:
        """Other assets this one depends on (e.g. IR assets for FX)."""
        return dict(self._dependencies)

    @property
    def n_scenarios(self) -> int:
        """Number of shock scenarios."""
        if self._shocks is not None and hasattr(self._shocks, "n_scenarios"):
            return self._shocks.n_scenarios
        return 0

    # ── Loading orchestration ─────────────────────────────────────────

    def load(self, api_client: Any) -> None:
        """Load all dependencies, market data, and shocks.

        Runs the three hooks in order, then validates:
        1. _load_dependencies() — other Assets this one needs
        2. _load()              — all market data for this asset class
        3. _load_shocks()       — typed shock scenarios
        4. validate()           — consistency checks
        """
        logger.info("Loading asset: %s (%s)", self.asset_name, self.asset_class)

        self._dependencies = self._load_dependencies(api_client)
        logger.debug("  dependencies: %s", list(self._dependencies.keys()))

        self._load(api_client)
        logger.debug("  market data loaded — spot: %s", self.spot)

        self._load_shocks(api_client)
        logger.debug("  shocks loaded: %s", type(self._shocks).__name__)

        if not self.validate():
            raise ValueError(f"Validation failed for {self.asset_name}")

        self._loaded = True
        logger.info(
            "Asset loaded: %s — %d scenarios",
            self.asset_name, self.n_scenarios,
        )

    # ── Abstract hooks (subclass implements) ──────────────────────────

    @abstractmethod
    def _load_dependencies(self, api_client: Any) -> Dict[str, "Asset"]:
        """Load and return dependent assets.

        FX loads domestic and foreign IR assets.
        IR returns ``{}`` (leaf node).
        """
        ...

    @abstractmethod
    def _load(self, api_client: Any) -> None:
        """Load all market data for this asset class.

        This single hook covers everything: spot, time series, curves,
        vol surfaces/cubes, forward points, dividends — whatever this
        asset class needs.  The subclass decides the internal order.

        Dependencies are already loaded when this runs.

        Must set at minimum: self.spot, self.spot_series
        Plus any asset-class-specific fields (vol_surface, curve, etc.)
        """
        ...

    @abstractmethod
    def _load_shocks(self, api_client: Any) -> None:
        """Load shock scenarios into a typed shock dataclass.

        Runs after _load(), so all market data grids (curve tenors,
        vol expiries, etc.) are available for axis alignment.

        Must set: self._shocks (FXShocks, IRShocks, etc.)
        """
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Validate loaded data.  Return False on failure.

        Should check:
          - spot and spot_series present
          - Asset-specific data present (curve, vol, etc.)
          - Shock shapes internally consistent (shocks.validate_shapes())
          - Shock grids aligned with market data grids
        """
        ...

    # ── Repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "empty"
        return (
            f"{self.__class__.__name__}("
            f"name={self.asset_name!r}, "
            f"class={self.asset_class!r}, "
            f"status={status})"
        )
