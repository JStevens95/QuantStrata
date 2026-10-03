"""
Base asste class - self-contained market data + shocks for one risk factors.

After load(), an Asset holds everything the pipeline needs:
    - Spot price + historical time series
    - Volatility surface (asset-class-specific shape).
    - Dependent assets (e.g. IR curves required for FX pricing)
    - Shock scenarios per risk factor name.

Design priciples:
    - Asset does not manage file paths or I/O directly. An external loader API client is passed to load().
    - Asset-class specific fields (domestic_curves, vol_Type) live on subclasses, not the base.
    - Immutable config separated from mutable market state.
    - Dependencies are full Asset instances, not raw dicts - so an FXAsset's domestic_curve is an IRAsset
"""
from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from typing import Dict, Optional, Any

from src.static_replication.assets.types import AssetConfig

# define module level logging.
logger = logging.getLogger(__name__)


class Asset(ABC):
    """
    Abstract base: self.-contained market data + shocks for one risk factor.

    Lifecycle:
        1. Construct with AssetConfig (immutable identity).
        2. Call load(api_client) - fetches and populates all market data, dependent asset and shock scenarios.
        3. Pipeline reads .risk_factors, .shocks, dependent assets.

    Subclasses must imple,ent four protected methods that defines what to load and how to assemble it:
        _load() - market data (spot, curves, vol surface etc.)
        _load_shocks() - scenario arrays per risk factor.
        _load_dependencies() - other Asset instances this one needs.

    The public load() method calls these in the correct order:
        - dependencies first, then own market data, then shocks, then validation.
    """

    def __init__(self, config: AssetConfig) -> None:
        """Initiate Asset instances."""

        # define required variables.
        self.config = config

        # market data (populated by load)
        self.spot: Optional[float] = None

        # shocks: backing store - use self._shocks to avoid recursion with @property shocks.
        self._shocks: Any = {}

        # dependent assets: {asset_name: Asset instance}
        self.dependencies: Dict[str, Asset] = {}

        # flag for loaded data
        self._loaded: bool = False

    @property
    def asset_class(self) -> str:
        """Return asset class."""
        return self.config.asset_class

    @property
    def asset_name(self) -> str:
        """Return asset name."""
        return self.config.asset_name

    @property
    def factor_id(self) -> str:
        """Return factor id."""
        return self.config.factor_id

    @property
    def _is_loaded(self) -> bool:
        """Return True if Asset instance is loaded."""
        return self._loaded

    @property
    def shocks(self) -> Any:
        """Return shock scenarios."""
        return self._shocks

    @property
    def dependent_assets(self) -> Dict[str, Asset]:
        """Return dependent assets."""
        return dict(self.dependencies)

    @property
    def n_scenarios(self) -> int:
        """Return number of scenarios."""
        if not self._shocks:
            return 0
        # typed shock objects (IrShocks / FxShocks) expose n_scenarios()
        if hasattr(self._shocks, "n_scenarios"):
            return self._shocks.n_scenarios
        # legacy dict path: {rf_id: np.ndarray}
        first = next(iter(self._shocks.values()))
        return first.shape[0]

    def load(self, api_client: Any) -> None:
        """
        Load all market data, dependencies and shocks.

        Calls for four abstract hooks in the correct order:
        1. Dependencies first (they may be needed for vol construction).
        2. Spot price + time series
        3. Volatility surface (may use dependent curve).
        4. Shock scenarios.
        5. Validation

        :param api_client:
        :return:
        """
        logger.info("Loading asst: %s (%s)", self.asset_name, self.asset_class)

        # load asset dependencies.
        self.dependencies = self._load_dependencies(api_client)
        logger.debug(" dependencies loaded: %s", list(self.dependencies.keys()))

        # load spot market data.
        self._load(api_client)
        logger.debug(" market data loaded: spot=%s", self.spot)

        # load shock scenarios.
        self._load_shocks(api_client)
        shocks_repr = list(self._shocks.keys()) if isinstance(self._shocks, dict) else type(self._shocks).__name__
        logger.debug(" shocks loaded: %s", shocks_repr)

        # validation - validate() raises on failure, return None on sucess.
        self.validate()

        # update loaded flag
        self._loaded = True
        logger.info("Asset loaded: %s - %d scenarios", self.asset_name, self.n_scenarios)

    @abstractmethod
    def _load_dependencies(self, api_client: Any) -> Dict[str, Asset]:
        """
        Load and return dependent assets.

        For example, FX loads domestic and foreign IR curves. Each dependency is a fully loaded assset instance.

        :param api_client:
        :return:
        """
        ...

    @abstractmethod
    def _load(self, api_client: Any) -> None:
        """
        Load all market data for this asset.

        This single hook covers everything; spot, timeseries, curves, vol surface/cube, forward points, dividends -
        whatever else the asset needs. The subclass decides the internal order.

        Dependencies are already loaded at this point.

        Must set at minimum; self._spot, plus any asset class specific fields.

        :param api_client:
        :return:
        """
        ...

    @abstractmethod
    def _load_shocks(self, api_client: Any) -> None:
        """
        Load shock scenario arrays for own risk factors.

        Must set ``self.shocks``
        :param api_client:
        :return:
        """
        ...

    @abstractmethod
    def validate(self) -> None:
        """
        Validate loaded data.

        Raise ValidationError on failure. Checks: spot not None/NaN, vol surface shape correct, shock shape consistent,
        dependencies loaded etc.
        :return:
        """
        ...

    def __repr__(self) -> str:
        """Return a concise string representation showing key info and loaded state."""
        status = "loaded" if self._loaded else "not loaded"
        return f"{self.__class__.__name__}(name={self.asset_name!r}, class={self.asset_class!r}, status={status!r})"
