# rade_sr — Static Replication Skeleton (Full Source)

> This document contains the complete source code for all 52 Python files
> in the `src/rade_sr/` skeleton. Copy each file to your work environment.
> Search for `TODO(wire)` to find the integration points where your
> existing code should be connected.

## Table of Contents

### (root)/
1. [__init__.py](#--init---py)

### api/
2. [api/__init__.py](#api---init---py)
3. [api/client.py](#api-client-py)

### config/
4. [config/__init__.py](#config---init---py)
5. [config/loader.py](#config-loader-py)

### core/
6. [core/__init__.py](#core---init---py)
7. [core/exceptions.py](#core-exceptions-py)
8. [core/protocols.py](#core-protocols-py)
9. [core/types.py](#core-types-py)

### data/
10. [data/__init__.py](#data---init---py)
11. [data/graph.py](#data-graph-py)
12. [data/interpolation.py](#data-interpolation-py)
13. [data/manager.py](#data-manager-py)
14. [data/standardiser.py](#data-standardiser-py)

### instruments/
15. [instruments/__init__.py](#instruments---init---py)
16. [instruments/base.py](#instruments-base-py)
17. [instruments/fx.py](#instruments-fx-py)
18. [instruments/rates.py](#instruments-rates-py)

### market_data/
19. [market_data/__init__.py](#market-data---init---py)
20. [market_data/manager.py](#market-data-manager-py)
21. [market_data/scenarios.py](#market-data-scenarios-py)

### orchestration/
22. [orchestration/__init__.py](#orchestration---init---py)
23. [orchestration/orchestrator.py](#orchestration-orchestrator-py)

### portfolio/
24. [portfolio/__init__.py](#portfolio---init---py)
25. [portfolio/manager.py](#portfolio-manager-py)

### pricing/
26. [pricing/__init__.py](#pricing---init---py)
27. [pricing/option_pricer.py](#pricing-option-pricer-py)

### replication/
28. [replication/__init__.py](#replication---init---py)
29. [replication/clustering/__init__.py](#replication-clustering---init---py)
30. [replication/clustering/manager.py](#replication-clustering-manager-py)
31. [replication/encoding/__init__.py](#replication-encoding---init---py)
32. [replication/encoding/trade_encoder.py](#replication-encoding-trade-encoder-py)
33. [replication/preprocessing/__init__.py](#replication-preprocessing---init---py)
34. [replication/preprocessing/pipeline.py](#replication-preprocessing-pipeline-py)
35. [replication/preprocessing/registry.py](#replication-preprocessing-registry-py)
36. [replication/preprocessing/stages/__init__.py](#replication-preprocessing-stages---init---py)
37. [replication/preprocessing/stages/asset_builders.py](#replication-preprocessing-stages-asset-builders-py)
38. [replication/preprocessing/stages/cluster_resolver.py](#replication-preprocessing-stages-cluster-resolver-py)
39. [replication/preprocessing/stages/path_resolver.py](#replication-preprocessing-stages-path-resolver-py)
40. [replication/preprocessing/stages/pnl_engine.py](#replication-preprocessing-stages-pnl-engine-py)
41. [replication/preprocessing/stages/trade_generator.py](#replication-preprocessing-stages-trade-generator-py)
42. [replication/reduction/__init__.py](#replication-reduction---init---py)
43. [replication/reduction/base.py](#replication-reduction-base-py)
44. [replication/reduction/grid.py](#replication-reduction-grid-py)
45. [replication/reduction/kpca.py](#replication-reduction-kpca-py)
46. [replication/reduction/trade_reducer.py](#replication-reduction-trade-reducer-py)
47. [replication/trades/__init__.py](#replication-trades---init---py)
48. [replication/trades/generator.py](#replication-trades-generator-py)
49. [replication/trades/pnl.py](#replication-trades-pnl-py)

### utils/
50. [utils/__init__.py](#utils---init---py)
51. [utils/mapping.py](#utils-mapping-py)
52. [utils/tools.py](#utils-tools-py)

---

## `__init__.py`

**Path:** `src/rade_sr/__init__.py`

```python
"""
Static Replication (rade_sr) — preprocessing library for ML-based trade replication.

Transforms raw portfolio data, market data, and scenarios into structured
ReplicationJob objects that feed downstream ML pipelines (rade_ml_pt).

Architecture follows strict layered dependencies — see docs/rade_sr_skeleton.md.
"""
```

---

## `api/__init__.py`

**Path:** `src/rade_sr/api/__init__.py`

```python
"""API client layer — internal API boundaries (thin)."""
```

---

## `api/client.py`

**Path:** `src/rade_sr/api/client.py`

```python
"""
Internal API client — thin wrapper for portfolio and market data APIs.

Handles HTTP transport, authentication, pagination, retries, and rate
limiting. Business logic does NOT belong here — this is purely an
adapter that translates API responses into Python objects.

TODO(wire): Replace the method bodies with calls to your internal API.
The exact endpoints, authentication, and response formats depend on
your firm's infrastructure.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class InternalAPIClient:
    """Client for internal portfolio and market data APIs.

    Parameters
    ----------
    base_url : str
        API base URL.
    auth_token : str or None
        Authentication token. If None, assumed to be handled by
        environment or middleware.
    timeout : int
        Request timeout in seconds.

    TODO(wire): Adapt constructor to your authentication mechanism
    (OAuth, API key, Kerberos, etc.).
    """

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._timeout = timeout

    # ── Portfolio API ─────────────────────────────────────────────────

    def get_target_portfolio(
        self,
        portfolio_id: str,
        as_of: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch the target portfolio from the internal API.

        Parameters
        ----------
        portfolio_id : str
            Portfolio identifier.
        as_of : str or None
            As-of date (ISO format). None = latest.

        Returns
        -------
        list[dict]
            List of trade records. Each dict should contain at least:
            ``trade_id``, ``desk``, ``product_type``, ``ccy``,
            ``notional``, and any other attributes used for clustering.

        TODO(wire): Replace with your actual API call:
            response = requests.get(
                f"{self._base_url}/portfolio/{portfolio_id}",
                params={"as_of": as_of},
                headers=self._auth_headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()["trades"]
        """
        raise NotImplementedError("Wire to your portfolio API")

    # ── Market Data API ───────────────────────────────────────────────

    def get_fx_spot(self, ccy_pair: str, as_of: Optional[str] = None) -> float:
        """Fetch FX spot rate.

        TODO(wire): Replace with your market data API call.
        """
        raise NotImplementedError("Wire to your FX spot API")

    def get_fx_forwards(
        self, ccy_pair: str, as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch FX forward curve.

        TODO(wire): Replace with your FX forward curve API.
        """
        raise NotImplementedError("Wire to your FX forwards API")

    def get_fx_vol_surface(
        self, ccy_pair: str, as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch FX vol surface.

        TODO(wire): Replace with your FX vol surface API.
        Expected return: dict with tenors, strikes, vol_matrix, or
        your existing vol surface object.
        """
        raise NotImplementedError("Wire to your FX vol surface API")

    def get_discount_curve(
        self, currency: str, as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch discount curve.

        TODO(wire): Replace with your IR curve API.
        """
        raise NotImplementedError("Wire to your discount curve API")

    def get_projection_curve(
        self, currency: str, as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch projection (forward) curve.

        TODO(wire): Replace with your projection curve API.
        """
        raise NotImplementedError("Wire to your projection curve API")

    def get_swaption_vol(
        self, currency: str, as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch swaption vol cube.

        TODO(wire): Replace with your swaption vol API.
        """
        raise NotImplementedError("Wire to your swaption vol API")

    # ── Shock / Scenario API ──────────────────────────────────────────

    def get_fx_shocks(
        self, ccy_pair: str, as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch FX shock scenarios.

        TODO(wire): Replace with your shock/scenario API.
        """
        raise NotImplementedError("Wire to your FX shocks API")

    def get_ir_shocks(
        self, currency: str, as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch IR shock scenarios.

        TODO(wire): Replace with your IR shocks API.
        """
        raise NotImplementedError("Wire to your IR shocks API")

    # ── Helpers ───────────────────────────────────────────────────────

    def _auth_headers(self) -> Dict[str, str]:
        """Build authentication headers.

        TODO(wire): Adapt to your auth mechanism.
        """
        headers: Dict[str, str] = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers
```

---

## `config/__init__.py`

**Path:** `src/rade_sr/config/__init__.py`

```python
"""Configuration loading and validation."""
```

---

## `config/loader.py`

**Path:** `src/rade_sr/config/loader.py`

```python
"""
Configuration loader — reads YAML/CSV configs and validates against dataclasses.

Provides a single entry point for loading the pipeline's configuration
from disk, validating required keys, and constructing PipelineConfig.

TODO(wire): Adapt the YAML keys and validation rules to match your
actual configuration files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.rade_sr.core.exceptions import ConfigurationError
from src.rade_sr.replication.preprocessing.pipeline import PipelineConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates pipeline configuration from YAML files.

    Parameters
    ----------
    config_dir : str or Path
        Directory containing configuration files.

    Expected files:
        - ``config_model.yaml``: Pipeline, clustering, and trade generation config.
        - ``config_api.yaml``: API endpoints and authentication.
        - ``mapping.csv``: Asset class / factor ID mappings (optional).
    """

    def __init__(self, config_dir: Union[str, Path]) -> None:
        self._dir = Path(config_dir)

    def load_pipeline_config(
        self,
        model_config_file: str = "config_model.yaml",
        api_config_file: str = "config_api.yaml",
    ) -> PipelineConfig:
        """Load and validate the full pipeline configuration.

        Returns
        -------
        PipelineConfig
            Validated configuration ready for PreprocessingPipeline.run().

        Raises
        ------
        ConfigurationError
            If required keys are missing or values are invalid.

        TODO(wire): Adjust the YAML parsing below to match your actual
        config file structure.
        """
        model_cfg = self._load_yaml(self._dir / model_config_file)
        api_cfg = self._load_yaml(self._dir / api_config_file)

        self._validate_model_config(model_cfg)

        asset_config = self._build_asset_config(model_cfg, api_cfg)

        return PipelineConfig(
            cluster_config=model_cfg.get("clustering", {}),
            path_config=model_cfg.get("paths", {}),
            asset_config=asset_config,
            trade_config=model_cfg.get("trade_generation", {}),
            max_workers=model_cfg.get("max_workers", 4),
            fail_fast=model_cfg.get("fail_fast", True),
        )

    def load_factor_mapping(
        self,
        mapping_file: str = "mapping.csv",
    ) -> Dict[str, Dict[str, str]]:
        """Load factor ID to asset class mapping from CSV.

        Returns
        -------
        dict
            ``{factor_id: {asset_class, pair/currency, ...}}``.

        TODO(wire): Adapt CSV columns to your mapping file format.
        """
        import pandas as pd
        path = self._dir / mapping_file
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        return {
            row["factor_id"]: row.to_dict()
            for _, row in df.iterrows()
        }

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        import yaml
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigurationError(f"Config file must be a dict: {path}")
        return data

    @staticmethod
    def _validate_model_config(cfg: Dict[str, Any]) -> None:
        """Validate required top-level keys in model config.

        TODO(wire): Add validation rules specific to your config schema.
        """
        required = ["clustering", "paths"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ConfigurationError(
                f"Missing required config keys: {missing}"
            )

    @staticmethod
    def _build_asset_config(
        model_cfg: Dict[str, Any],
        api_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge model and API config into per-factor asset config.

        TODO(wire): Adapt to your config structure. This should produce
        the asset_config dict expected by PipelineConfig, where each
        factor_id maps to its asset_class and loading parameters.
        """
        factors = model_cfg.get("risk_factors", {})
        api_base = api_cfg.get("base_url", "")

        asset_config: Dict[str, Any] = {}
        for factor_id, factor_cfg in factors.items():
            merged = dict(factor_cfg)
            merged["api_base_url"] = api_base
            asset_config[factor_id] = merged

        return asset_config
```

---

## `core/__init__.py`

**Path:** `src/rade_sr/core/__init__.py`

```python
"""
Core layer — innermost. Types, protocols, and exceptions.

This module has ZERO imports from any other rade_sr layer.
If a type is needed here that currently lives elsewhere, move it here.
"""
from src.rade_sr.core.types import (
    ClusterPaths,
    RiskFactor,
    ElementaryTrade,
    TradeGenerationContext,
    ElementaryTradePnL,
    IntermediateClusterData,
    ReplicationJob,
)
from src.rade_sr.core.protocols import (
    ClusterResolver,
    PathResolver,
    RiskFactorBuilder,
    TradeGenerator,
    PnLEngine,
)
from src.rade_sr.core.exceptions import (
    StaticReplicationError,
    UnknownAssetClassError,
    PipelineError,
    PnLComputationError,
)

__all__ = [
    "ClusterPaths",
    "RiskFactor",
    "ElementaryTrade",
    "TradeGenerationContext",
    "ElementaryTradePnL",
    "IntermediateClusterData",
    "ReplicationJob",
    "ClusterResolver",
    "PathResolver",
    "RiskFactorBuilder",
    "TradeGenerator",
    "PnLEngine",
    "StaticReplicationError",
    "UnknownAssetClassError",
    "PipelineError",
    "PnLComputationError",
]
```

---

## `core/exceptions.py`

**Path:** `src/rade_sr/core/exceptions.py`

```python
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
```

---

## `core/protocols.py`

**Path:** `src/rade_sr/core/protocols.py`

```python
"""
Protocol definitions for all pipeline stage contracts.

Every cross-layer dependency points at these abstractions, not at
concrete classes. The PreprocessingPipeline depends on Protocol types
only — it has zero imports from instruments/, market_data/, or
replication/stages/.

When adding a new stage, define its Protocol here FIRST, then implement.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from src.rade_sr.core.types import (
    ClusterPaths,
    ElementaryTrade,
    ElementaryTradePnL,
    RiskFactor,
    TradeGenerationContext,
)


@runtime_checkable
class ClusterResolver(Protocol):
    """Resolves user configuration into a flat list of cluster IDs.

    Wraps whatever clustering strategy is in use (k-means, user-defined
    keys, hierarchical, etc.). The pipeline only sees cluster ID strings.
    """

    def resolve(self, config: Dict[str, Any]) -> List[str]:
        """Return ordered list of cluster ID strings from the given config.

        Parameters
        ----------
        config : dict
            Clustering configuration. Expected keys depend on implementation
            but typically include ``n_clusters``, ``method``, ``cluster_key``,
            ``cluster_key_values``, etc.
        """
        ...


@runtime_checkable
class PathResolver(Protocol):
    """Derives artifact filesystem paths for a given cluster.

    Different path layouts (convention-based, database-backed, cloud storage)
    can be swapped by providing a different PathResolver implementation.
    """

    def resolve(self, cluster_id: str, path_config: Dict[str, Any]) -> ClusterPaths:
        """Return the four canonical artifact paths for one cluster.

        Parameters
        ----------
        cluster_id : str
            The cluster to resolve paths for.
        path_config : dict
            Path configuration. Typically includes ``root`` directory and
            filename conventions for target PnL, target attributes,
            elementary PnL, elementary attributes.
        """
        ...


@runtime_checkable
class RiskFactorBuilder(Protocol):
    """Builds a fully loaded RiskFactor for one factor ID.

    Each asset class has its own implementation registered via the
    RiskFactorRegistry. The pipeline dispatches to the correct builder
    using ``RiskFactorRegistry.get(asset_class)``.

    Implementations must define ``asset_class`` as a class-level string
    attribute — this is the registry key.
    """

    asset_class: str

    def build(
        self,
        factor_id: str,
        factor_config: Dict[str, Any],
        cluster_id: str,
    ) -> RiskFactor:
        """Load market data + scenarios and return a fully assembled RiskFactor.

        Parameters
        ----------
        factor_id : str
            Unique factor identifier (e.g. ``"EURUSD"``, ``"EUR_6M"``).
        factor_config : dict
            Per-factor configuration from the pipeline config. Contains
            ``asset_class`` plus asset-specific keys (``pair``, ``currency``,
            ``underlying``, ``issuer``, etc.).
        cluster_id : str
            The cluster this factor is being built for (useful for metadata).
        """
        ...


@runtime_checkable
class TradeGenerator(Protocol):
    """Generates elementary trades by interrogating assembled risk factors.

    The generator examines each risk factor's market data (vol surfaces,
    rate curves, spot levels) to produce contextually appropriate trades
    with appropriate strikes, tenors, and payoff types.
    """

    def generate(self, context: TradeGenerationContext) -> List[ElementaryTrade]:
        """Generate and deduplicate elementary trades for one cluster.

        Parameters
        ----------
        context : TradeGenerationContext
            Contains the cluster ID, all assembled risk factors, and the
            user-provided trade configuration (notional, expiry grid, etc.).
        """
        ...


@runtime_checkable
class PnLEngine(Protocol):
    """Computes PnL for a batch of elementary trades across their risk factors.

    Operates at the portfolio level (Phase 2) to avoid redundant pricing
    of shared risk factors across clusters. Groups trades by factor_id
    and makes one vectorised pricer call per unique factor group.
    """

    def compute_batch(
        self,
        trades: List[ElementaryTrade],
        risk_factors: Dict[str, RiskFactor],
    ) -> Dict[str, ElementaryTradePnL]:
        """Compute PnL for all trades, grouped by risk factor.

        Parameters
        ----------
        trades : list[ElementaryTrade]
            Portfolio-wide list of all elementary trades (from all clusters).
        risk_factors : dict[str, RiskFactor]
            Deduplicated risk factors keyed by factor_id.

        Returns
        -------
        dict[str, ElementaryTradePnL]
            Mapping from trade_id to its PnL vector. Covers all input trades.
        """
        ...
```

---

## `core/types.py`

**Path:** `src/rade_sr/core/types.py`

```python
"""
Core domain types for the static replication library.

Every inter-stage handoff uses a typed dataclass from this module.
No raw dicts between stages — a typo on a dataclass attribute is caught
immediately by the IDE; a typo on a dict key fails silently at runtime.

This module must have ZERO imports from other rade_sr layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Stage 1 output: resolved filesystem paths for one cluster
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClusterPaths:
    """Immutable set of four artifact paths for a single cluster.

    These paths point to the on-disk locations where the pipeline will
    read target/elementary PnL and attribute data. Frozen so they cannot
    be accidentally mutated after resolution.
    """
    target_pnl: Path
    target_attributes: Path
    elem_pnl: Path
    elem_attributes: Path


# ─────────────────────────────────────────────────────────────────────
# Stage 2 output: assembled risk factor with market data + scenarios
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RiskFactor:
    """One risk factor fully loaded with market data and shock scenarios.

    Parameters
    ----------
    factor_id : str
        Unique identifier for this risk factor (e.g. ``"EURUSD"``, ``"EUR_6M"``).
    asset_class : str
        Asset class key used for registry dispatch (e.g. ``"fx"``, ``"rates"``,
        ``"eq"``, ``"cr"``).
    market_data : Any
        The loaded market object for this factor. Shape depends on asset class:
        - FX: spot, forward curve, vol surface (tenors × strikes)
        - IR: discount curve, projection curve, swaption vol cube
        - EQ: spot, dividend schedule, vol surface
        - CR: credit spread curve, recovery rate, default intensity

        TODO(wire): This will hold your existing MarketDataManager output
        for this factor. Keep the type as Any until you stabilise the
        per-asset-class market data objects, then narrow to a Union or
        generic.
    scenarios : Any
        Shock scenario set for this factor. Typically a numpy array or
        a structured dict of shifted market states.

        TODO(wire): This will hold your existing ShockManager / ScenarioManager
        output. Same advice as market_data re: type narrowing.
    metadata : dict
        Arbitrary metadata (pair name, currency, cluster origin, source, etc.).
    """
    factor_id: str
    asset_class: str
    market_data: Any
    scenarios: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Stage 3 output: one elementary replicating instrument
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ElementaryTrade:
    """One elementary (replicating) instrument in the trade universe.

    Parameters
    ----------
    trade_id : str
        Globally unique trade identifier. Typically encoded from instrument
        parameters + cluster + factor (e.g. ``"fx_EURUSD_call_1.10_3M_cluster_0"``).
    asset_class : str
        Asset class of the underlying risk factor.
    payoff_type : str
        Instrument payoff type (e.g. ``"call"``, ``"put"``, ``"swap"``,
        ``"digital"``, ``"forward"``).
    factor_id : str
        The risk factor this trade is priced against.
    parameters : dict
        Instrument-specific parameters. Examples:
        - FX option: ``{"strike": 1.10, "expiry": 0.25, "ccy_pair": "EURUSD"}``
        - IR swap: ``{"fixed_rate": 0.03, "tenor": 5.0, "currency": "EUR"}``
    notional : float
        Trade notional (default 1.0 for unit replication).
    cluster_id : str
        Cluster this trade was generated for (used for PnL distribution in Phase 2).
    """
    trade_id: str
    asset_class: str
    payoff_type: str
    factor_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    notional: float = 1.0
    cluster_id: str = ""


# ─────────────────────────────────────────────────────────────────────
# Context object passed to the trade generator
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TradeGenerationContext:
    """Everything the trade generator needs for one cluster.

    Bundles the cluster's assembled risk factors with the user-provided
    trade configuration so the generator can interrogate market data
    (vol surfaces, rate curves) to produce contextually appropriate
    strikes, tenors, and payoff types.
    """
    cluster_id: str
    risk_factors: Dict[str, RiskFactor]
    trade_config: Dict[str, Any]

    def get_factor(self, factor_id: str) -> RiskFactor:
        """Retrieve a specific risk factor by ID."""
        return self.risk_factors[factor_id]

    def factor_ids(self) -> List[str]:
        """All risk factor IDs available in this context."""
        return list(self.risk_factors)


# ─────────────────────────────────────────────────────────────────────
# PnL result for one elementary trade
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ElementaryTradePnL:
    """PnL vector for a single elementary trade across all scenarios.

    Parameters
    ----------
    trade_id : str
        Must match the corresponding ElementaryTrade.trade_id.
    factor_id : str
        The risk factor this PnL was computed against.
    cluster_id : str
        Originating cluster (for distribution in Phase 2).
    pnl_vector : np.ndarray
        Shape ``(n_scenarios,)`` — one PnL value per shock scenario.
    """
    trade_id: str
    factor_id: str
    cluster_id: str
    pnl_vector: np.ndarray


# ─────────────────────────────────────────────────────────────────────
# Phase 1 output: intermediate cluster data (no PnL yet)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IntermediateClusterData:
    """Phase 1 output for a single cluster.

    Contains resolved paths, assembled risk factors, and generated
    elementary trades. PnL has NOT been computed yet — that happens
    in Phase 2 at the portfolio level to avoid redundant pricing.
    """
    cluster_id: str
    cluster_info: ClusterPaths
    cluster_assets: Dict[str, RiskFactor]
    cluster_elem_trades: List[ElementaryTrade]

    def all_trade_ids(self) -> List[str]:
        """Flat list of trade IDs in generation order."""
        return [t.trade_id for t in self.cluster_elem_trades]

    @property
    def n_trades(self) -> int:
        """Number of elementary trades in this cluster."""
        return len(self.cluster_elem_trades)

    @property
    def n_factors(self) -> int:
        """Number of distinct risk factors in this cluster."""
        return len(self.cluster_assets)


# ─────────────────────────────────────────────────────────────────────
# Phase 2 output: fully assembled replication job
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ReplicationJob:
    """Final pipeline output for one cluster — what downstream ML consumes.

    Contains everything needed to feed ``rade_ml_pt`` data pipelines:
    paths to target data, the risk factor universe with market data and
    scenarios, the elementary trade universe, and the computed PnL matrix.

    The ``pnl_matrix()`` convenience method returns the standard
    ``(n_trades, n_scenarios)`` array that the regression solver expects,
    with row order matching ``cluster_elem_trades`` order.
    """
    cluster_id: str
    cluster_info: ClusterPaths
    cluster_assets: Dict[str, RiskFactor]
    cluster_elem_trades: List[ElementaryTrade]
    elem_trade_pnls: Dict[str, ElementaryTradePnL]

    def pnl_matrix(self) -> np.ndarray:
        """Build ``(n_trades, n_scenarios)`` matrix.

        Row order matches ``cluster_elem_trades`` order so that
        ``pnl_matrix()[i]`` corresponds to ``cluster_elem_trades[i]``.

        Raises
        ------
        KeyError
            If any trade in ``cluster_elem_trades`` is missing from
            ``elem_trade_pnls``.
        """
        return np.stack([
            self.elem_trade_pnls[t.trade_id].pnl_vector
            for t in self.cluster_elem_trades
        ])

    @property
    def n_trades(self) -> int:
        """Number of elementary trades."""
        return len(self.cluster_elem_trades)

    @property
    def n_scenarios(self) -> int:
        """Number of shock scenarios (from first trade's PnL vector)."""
        if not self.elem_trade_pnls:
            return 0
        first_pnl = next(iter(self.elem_trade_pnls.values()))
        return first_pnl.pnl_vector.shape[0]

    @property
    def trade_ids(self) -> List[str]:
        """Trade IDs in generation order."""
        return [t.trade_id for t in self.cluster_elem_trades]

    @property
    def factor_ids(self) -> List[str]:
        """Unique risk factor IDs in this cluster."""
        return list(self.cluster_assets.keys())

    def to_job_dict(self) -> Dict[str, Any]:
        """Convert to the legacy dict format expected by rade_ml_pt build_dataset.

        TODO(wire): Adapt the keys below to match the exact dict structure
        your hybrid_gnn_rnn build_dataset() expects under job["cluster_info"].
        The typical keys are: target_pnl_path, elementary_pnl_path,
        target_attribs_path, elementary_attribs_path.
        """
        return {
            "cluster_info": {
                "target_pnl_path": str(self.cluster_info.target_pnl),
                "target_attribs_path": str(self.cluster_info.target_attributes),
                "elementary_pnl_path": str(self.cluster_info.elem_pnl),
                "elementary_attribs_path": str(self.cluster_info.elem_attributes),
            },
            "cluster_id": self.cluster_id,
            "cluster_assets": self.cluster_assets,
            "n_trades": self.n_trades,
            "n_scenarios": self.n_scenarios,
        }
```

---

## `data/__init__.py`

**Path:** `src/rade_sr/data/__init__.py`

```python
"""
Data processing layer — loading, graph construction, interpolation, standardisation.

Provides utilities shared across the replication pipeline for data I/O,
trade graph building, surface/curve interpolation, and PnL standardisation.
"""
```

---

## `data/graph.py`

**Path:** `src/rade_sr/data/graph.py`

```python
"""
Trade graph construction utilities.

Builds the k-NN trade relationship graph from encoded trade attributes.
This graph feeds the GNN component of the hybrid GNN-RNN model.

TODO(wire): This mirrors the graph building logic in
``rade_ml_pt/data/hybrid_gnn_rnn/build.py::build_trade_graph()``.
If you want a single source of truth, consider importing from rade_ml_pt
instead of duplicating. This module exists so rade_sr can build graphs
independently if needed (e.g. for pre-computation before ML training).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def build_trade_graph(
    encoded_features: np.ndarray,
    k: int = 10,
    kernel: str = "rbf",
    sigma: float = 1.0,
    self_loops: bool = True,
) -> Dict[str, Any]:
    """Build a sparse k-NN adjacency graph from encoded trade features.

    Parameters
    ----------
    encoded_features : np.ndarray
        Feature matrix, shape ``(n_trades, n_features)``.
    k : int
        Number of nearest neighbours per node.
    kernel : str
        Edge weight kernel: ``"rbf"`` or ``"binary"``.
    sigma : float
        RBF kernel bandwidth (only used if kernel is ``"rbf"``).
    self_loops : bool
        Whether to include self-loops in the adjacency.

    Returns
    -------
    dict
        - ``sparse_indices``: np.ndarray, shape ``(n_edges, 2)`` — [row, col] pairs.
        - ``sparse_values``: np.ndarray, shape ``(n_edges,)`` — edge weights.
        - ``sparse_shape``: tuple ``(n_trades, n_trades)``.
        - ``dense_adjacency``: np.ndarray (optional, for debugging).

    TODO(wire): If your existing TradeGraphBuilder has a different API,
    adapt this function to call it. The output format must match what
    rade_ml_pt expects for ``static_inputs`` in the DataLoader.
    """
    n_trades = encoded_features.shape[0]
    k = min(k, n_trades - 1)

    distances = cdist(encoded_features, encoded_features, metric="euclidean")

    rows, cols, vals = [], [], []
    for i in range(n_trades):
        dists_i = distances[i]
        if not self_loops:
            dists_i[i] = np.inf
        neighbours = np.argsort(dists_i)[:k + (1 if self_loops else 0)]

        for j in neighbours:
            weight = _compute_weight(dists_i[j], kernel, sigma)
            rows.append(i)
            cols.append(j)
            vals.append(weight)

    sparse_indices = np.array(list(zip(rows, cols)), dtype=np.int64)
    sparse_values = np.array(vals, dtype=np.float32)

    return {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "sparse_shape": (n_trades, n_trades),
    }


def _compute_weight(distance: float, kernel: str, sigma: float) -> float:
    """Compute edge weight from distance."""
    if kernel == "rbf":
        return float(np.exp(-(distance ** 2) / (2 * sigma ** 2)))
    elif kernel == "binary":
        return 1.0
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")
```

---

## `data/interpolation.py`

**Path:** `src/rade_sr/data/interpolation.py`

```python
"""
Surface and curve interpolation utilities.

Provides helpers for interpolating vol surfaces, rate curves, and
other market data objects at arbitrary points. Used by trade generators
to derive strike/tenor grids and by pricers to look up vols/rates.

TODO(wire): Wire to your existing interpolation routines. If you already
have surface/curve interpolation in your market data objects (e.g.
``vol_surface.get_vol(strike, expiry)``), this module may be thin or
used only for special cases (extrapolation, grid coarsening for ML).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


def interpolate_surface(
    surface: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    method: str = "linear",
    bounds_error: bool = False,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """Interpolate a 2D surface (e.g. vol surface) at query points.

    Parameters
    ----------
    surface : np.ndarray
        Surface values, shape ``(len(x_axis), len(y_axis))``.
        Example: vol surface indexed by [expiry, strike].
    x_axis : np.ndarray
        First axis values (e.g. expiries/tenors).
    y_axis : np.ndarray
        Second axis values (e.g. strikes/moneyness).
    x_query, y_query : np.ndarray
        Query points. Must be broadcastable.
    method : str
        Interpolation method: ``"linear"``, ``"nearest"``, ``"cubic"``.
    bounds_error : bool
        Whether to raise on out-of-bounds queries.
    fill_value : float or None
        Value for out-of-bounds. None = nearest boundary value.

    Returns
    -------
    np.ndarray
        Interpolated values at query points.
    """
    interpolator = RegularGridInterpolator(
        (x_axis, y_axis),
        surface,
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
    points = np.column_stack([x_query.ravel(), y_query.ravel()])
    return interpolator(points).reshape(x_query.shape)


def interpolate_curve(
    curve_values: np.ndarray,
    curve_tenors: np.ndarray,
    query_tenors: np.ndarray,
    method: str = "linear",
    extrapolate: bool = True,
) -> np.ndarray:
    """Interpolate a 1D curve (e.g. discount curve, credit spread curve).

    Parameters
    ----------
    curve_values : np.ndarray
        Curve values at pillar points.
    curve_tenors : np.ndarray
        Pillar tenors (e.g. year fractions).
    query_tenors : np.ndarray
        Tenors to interpolate at.
    method : str
        ``"linear"``, ``"cubic"``, ``"log_linear"`` (for discount factors).
    extrapolate : bool
        Whether to allow flat extrapolation beyond curve boundaries.

    Returns
    -------
    np.ndarray
        Interpolated values.
    """
    fill_value = "extrapolate" if extrapolate else np.nan

    if method == "log_linear":
        log_values = np.log(np.clip(curve_values, 1e-12, None))
        interpolator = interp1d(
            curve_tenors, log_values, kind="linear", fill_value=fill_value,
        )
        return np.exp(interpolator(query_tenors))

    interpolator = interp1d(
        curve_tenors, curve_values, kind=method, fill_value=fill_value,
    )
    return interpolator(query_tenors)
```

---

## `data/manager.py`

**Path:** `src/rade_sr/data/manager.py`

```python
"""
Data I/O manager — loading and saving pipeline artifacts.

Handles reading/writing of PnL DataFrames, trade attribute dicts,
and intermediate pipeline outputs. Abstracts the serialisation format
(parquet, joblib, JSON) behind a consistent interface.

TODO(wire): Align file formats with your existing CacheLoader / data
storage conventions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from src.rade_sr.core.types import ClusterPaths

logger = logging.getLogger(__name__)


class DataManager:
    """Reads and writes pipeline artifacts from/to disk.

    Parameters
    ----------
    base_dir : str or Path
        Root directory for all data artifacts.
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self._base_dir = Path(base_dir)

    def load_cluster_data(self, paths: ClusterPaths) -> Dict[str, Any]:
        """Load all four artifacts for a cluster.

        Returns
        -------
        dict
            Keys: ``target_pnl``, ``elementary_pnl``, ``target_attribs``,
            ``elementary_attribs``.

        TODO(wire): Adapt the loading calls below to your serialisation
        format. If you use CacheLoader, replace with:
            CacheLoader.load(file_path=str(paths.target_pnl))
        """
        return {
            "target_pnl": self._load_dataframe(paths.target_pnl),
            "elementary_pnl": self._load_dataframe(paths.elem_pnl),
            "target_attribs": self._load_attributes(paths.target_attributes),
            "elementary_attribs": self._load_attributes(paths.elem_attributes),
        }

    def save_cluster_data(
        self,
        paths: ClusterPaths,
        target_pnl: pd.DataFrame,
        elementary_pnl: pd.DataFrame,
        target_attribs: Dict[str, Any],
        elementary_attribs: Dict[str, Any],
    ) -> None:
        """Persist all four artifacts for a cluster.

        TODO(wire): Adapt to your serialisation format.
        """
        paths.target_pnl.parent.mkdir(parents=True, exist_ok=True)
        self._save_dataframe(target_pnl, paths.target_pnl)
        self._save_dataframe(elementary_pnl, paths.elem_pnl)
        self._save_attributes(target_attribs, paths.target_attributes)
        self._save_attributes(elementary_attribs, paths.elem_attributes)
        logger.info("Saved cluster data to %s", paths.target_pnl.parent)

    # ── Internal I/O helpers ──────────────────────────────────────────

    @staticmethod
    def _load_dataframe(path: Path) -> pd.DataFrame:
        """Load a DataFrame from parquet or CSV.

        TODO(wire): Match your file format. If you store as .pkl or .h5,
        change the reader accordingly.
        """
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path, index_col=0)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    @staticmethod
    def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
        """Save a DataFrame to parquet."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    @staticmethod
    def _load_attributes(path: Path) -> Dict[str, Any]:
        """Load trade attributes from parquet, CSV, or JSON.

        TODO(wire): Match your attribute storage format.
        """
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            return {col: df[col].tolist() for col in df.columns}
        elif path.suffix == ".json":
            import json
            with open(path, "r") as f:
                return json.load(f)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            return {col: df[col].tolist() for col in df.columns}
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    @staticmethod
    def _save_attributes(attribs: Dict[str, Any], path: Path) -> None:
        """Save trade attributes as parquet."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(attribs)
        df.to_parquet(path)
```

---

## `data/standardiser.py`

**Path:** `src/rade_sr/data/standardiser.py`

```python
"""
PnL standardisation utilities.

Provides the same transform types used in rade_ml_pt
(standard, robust, signed_log, signed_log_robust) so that
elementary PnL can be pre-standardised before passing to
the ML pipeline, or standardised identically during inference.

TODO(wire): If you prefer a single source of truth, import
directly from ``src.rade_ml_pt.features.transforms.standardiser``
instead of duplicating here. This module exists so rade_sr can
operate independently of rade_ml_pt when needed.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline


class SignedLogTransformer:
    """Sign-preserving log compression: ``sign(x) * log1p(|x|)``.

    Stateless transformer compatible with sklearn Pipeline.
    """

    def fit(self, X: np.ndarray, y: Any = None) -> SignedLogTransformer:
        """No-op fit (stateless). Returns self for Pipeline compatibility."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply signed log compression."""
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert signed log compression."""
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.expm1(np.abs(X))


def get_transformer(transform_type: str) -> Any:
    """Factory for PnL transformers.

    Parameters
    ----------
    transform_type : str
        One of: ``"standard"``, ``"robust"``, ``"signed_log"``,
        ``"signed_log_robust"``.

    Returns
    -------
    sklearn estimator or Pipeline
        A fitted-on-train-data transformer with ``.transform()``
        and ``.inverse_transform()`` methods.

    TODO(wire): If rade_ml_pt is available, you can import directly:
        from src.rade_ml_pt.features.transforms.standardiser import get_transformer
    """
    if transform_type == "standard":
        return StandardScaler()
    elif transform_type == "robust":
        return RobustScaler()
    elif transform_type == "signed_log":
        return SignedLogTransformer()
    elif transform_type == "signed_log_robust":
        return Pipeline([
            ("signed_log", SignedLogTransformer()),
            ("robust", RobustScaler()),
        ])
    else:
        raise ValueError(
            f"Unknown transform_type={transform_type!r}. "
            f"Supported: standard, robust, signed_log, signed_log_robust"
        )
```

---

## `instruments/__init__.py`

**Path:** `src/rade_sr/instruments/__init__.py`

```python
"""
Instrument definitions — product-level abstractions.

Defines the canonical representation of financial instruments across
asset classes. Used by trade generators and pricers to understand
payoff structure, exercise conventions, and parameter requirements.
"""
```

---

## `instruments/base.py`

**Path:** `src/rade_sr/instruments/base.py`

```python
"""
Base instrument abstractions.

All concrete instrument types (FX, IR, EQ, CR) inherit from
InstrumentSpec. This gives the trade generator and pricer a uniform
interface for extracting parameters regardless of asset class.
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
    common interface that pricers and generators depend on.
    """
    asset_class: str
    payoff_type: str
    factor_id: str

    @abstractmethod
    def to_pricer_params(self) -> Dict[str, Any]:
        """Extract the parameter dict that the pricer needs.

        Returns a flat dict of numeric/string values that can be
        vectorised across a batch of trades. Keys must be stable
        across all instances of the same payoff_type.

        TODO(wire): The exact keys depend on your OptionPricer's
        ``extract_params()`` interface. Match them here.
        """
        ...

    @abstractmethod
    def to_trade_id(self, cluster_id: str) -> str:
        """Generate a deterministic, globally unique trade ID.

        The ID must be stable across runs for the same instrument
        parameters so that deduplication works correctly.
        """
        ...


@dataclass
class StrikeGrid:
    """Defines a grid of strikes for option generation.

    TODO(wire): Adapt fields to match your existing strike grid
    conventions. Common patterns:
    - Absolute strikes: [1.05, 1.10, 1.15, ...]
    - Delta strikes: [10D, 25D, ATM, 25D, 10D]
    - Moneyness: [0.90, 0.95, 1.00, 1.05, 1.10]
    """
    values: List[float] = field(default_factory=list)
    convention: str = "absolute"
    reference_level: float = 0.0

    @classmethod
    def from_vol_surface(
        cls,
        vol_surface: Any,
        n_strikes: int = 11,
        wing_width: float = 0.15,
    ) -> StrikeGrid:
        """Derive a strike grid from a vol surface's ATM and wings.

        TODO(wire): This should interrogate the vol surface object to
        extract ATM level and produce a symmetric grid around it.
        The exact API depends on your vol surface representation.

        Parameters
        ----------
        vol_surface : Any
            Your market data vol surface object.
        n_strikes : int
            Number of strike points (odd number for symmetry around ATM).
        wing_width : float
            How far from ATM the wings extend (as fraction of ATM).
        """
        raise NotImplementedError("Wire to your vol surface's ATM extraction")


@dataclass
class TenorGrid:
    """Defines a grid of tenors/expiries for instrument generation.

    TODO(wire): Adapt to your tenor conventions. Common representations:
    - Year fractions: [0.25, 0.5, 1.0, 2.0, 5.0]
    - Period codes: ["3M", "6M", "1Y", "2Y", "5Y"]
    """
    values: List[float] = field(default_factory=list)
    convention: str = "year_fraction"

    @classmethod
    def from_curve(
        cls,
        curve: Any,
        max_tenor: float = 10.0,
    ) -> TenorGrid:
        """Derive a tenor grid from a rate curve's liquid points.

        TODO(wire): Interrogate the curve object to find its pillar
        tenors and return the liquid subset up to max_tenor.

        Parameters
        ----------
        curve : Any
            Your market data curve object.
        max_tenor : float
            Maximum tenor in years.
        """
        raise NotImplementedError("Wire to your curve's pillar extraction")
```

---

## `instruments/fx.py`

**Path:** `src/rade_sr/instruments/fx.py`

```python
"""
FX instrument definitions.

Covers the elementary instrument types used in FX static replication:
spot, forward, vanilla options, and (optionally) digitals and barriers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rade_sr.instruments.base import InstrumentSpec, StrikeGrid, TenorGrid


@dataclass
class FXForward(InstrumentSpec):
    """FX forward contract."""
    asset_class: str = "fx"
    payoff_type: str = "forward"
    factor_id: str = ""

    ccy_pair: str = ""
    expiry: float = 0.0
    forward_rate: float = 0.0
    notional: float = 1.0

    def to_pricer_params(self) -> Dict[str, Any]:
        """TODO(wire): Match keys to your pricer's forward pricing API."""
        return {
            "ccy_pair": self.ccy_pair,
            "expiry": self.expiry,
            "forward_rate": self.forward_rate,
            "notional": self.notional,
            "payoff_type": self.payoff_type,
        }

    def to_trade_id(self, cluster_id: str) -> str:
        return f"fx_{self.ccy_pair}_fwd_{self.expiry}_{cluster_id}"


@dataclass
class FXVanillaOption(InstrumentSpec):
    """FX European vanilla option (call or put)."""
    asset_class: str = "fx"
    payoff_type: str = "call"
    factor_id: str = ""

    ccy_pair: str = ""
    strike: float = 0.0
    expiry: float = 0.0
    notional: float = 1.0
    is_call: bool = True

    def to_pricer_params(self) -> Dict[str, Any]:
        """TODO(wire): Match keys to your pricer's option pricing API.

        Typical pricer inputs: spot, strike, expiry, vol, rate_dom,
        rate_for, is_call. Some of these come from the RiskFactor's
        market_data at pricing time, not from the instrument itself.
        """
        return {
            "ccy_pair": self.ccy_pair,
            "strike": self.strike,
            "expiry": self.expiry,
            "notional": self.notional,
            "is_call": self.is_call,
            "payoff_type": self.payoff_type,
        }

    def to_trade_id(self, cluster_id: str) -> str:
        side = "call" if self.is_call else "put"
        return f"fx_{self.ccy_pair}_{side}_{self.strike}_{self.expiry}_{cluster_id}"


@dataclass
class FXDigital(InstrumentSpec):
    """FX cash-or-nothing digital option."""
    asset_class: str = "fx"
    payoff_type: str = "digital"
    factor_id: str = ""

    ccy_pair: str = ""
    strike: float = 0.0
    expiry: float = 0.0
    payout: float = 1.0
    is_call: bool = True

    def to_pricer_params(self) -> Dict[str, Any]:
        """TODO(wire): Match to your digital pricer interface."""
        return {
            "ccy_pair": self.ccy_pair,
            "strike": self.strike,
            "expiry": self.expiry,
            "payout": self.payout,
            "is_call": self.is_call,
            "payoff_type": self.payoff_type,
        }

    def to_trade_id(self, cluster_id: str) -> str:
        side = "call" if self.is_call else "put"
        return f"fx_{self.ccy_pair}_digital_{side}_{self.strike}_{self.expiry}_{cluster_id}"


def generate_fx_elementary_trades(
    ccy_pair: str,
    factor_id: str,
    cluster_id: str,
    strike_grid: StrikeGrid,
    tenor_grid: TenorGrid,
    payoff_types: Optional[List[str]] = None,
    notional: float = 1.0,
) -> List[InstrumentSpec]:
    """Generate the full grid of FX elementary instruments.

    Produces calls and puts at every (strike, tenor) combination,
    plus optionally forwards and digitals.

    TODO(wire): This function is called by the RiskFactorAwareTradeGenerator
    for FX risk factors. Adjust payoff_types and grid logic to match your
    existing trade generation conventions.

    Parameters
    ----------
    ccy_pair : str
        Currency pair (e.g. ``"EURUSD"``).
    factor_id : str
        Risk factor identifier.
    cluster_id : str
        Originating cluster.
    strike_grid : StrikeGrid
        Grid of strikes (derived from vol surface).
    tenor_grid : TenorGrid
        Grid of expiries (from config or derived from curve).
    payoff_types : list[str] or None
        Which instrument types to generate. Default: ``["call", "put"]``.
    notional : float
        Per-trade notional.
    """
    payoff_types = payoff_types or ["call", "put"]
    trades: List[InstrumentSpec] = []

    for expiry in tenor_grid.values:
        for strike in strike_grid.values:
            if "call" in payoff_types:
                trades.append(FXVanillaOption(
                    factor_id=factor_id, ccy_pair=ccy_pair,
                    strike=strike, expiry=expiry, notional=notional,
                    is_call=True,
                ))
            if "put" in payoff_types:
                trades.append(FXVanillaOption(
                    factor_id=factor_id, ccy_pair=ccy_pair,
                    strike=strike, expiry=expiry, notional=notional,
                    is_call=False,
                ))
            if "digital" in payoff_types:
                trades.append(FXDigital(
                    factor_id=factor_id, ccy_pair=ccy_pair,
                    strike=strike, expiry=expiry,
                    is_call=True,
                ))

        if "forward" in payoff_types:
            trades.append(FXForward(
                factor_id=factor_id, ccy_pair=ccy_pair,
                expiry=expiry, notional=notional,
            ))

    return trades
```

---

## `instruments/rates.py`

**Path:** `src/rade_sr/instruments/rates.py`

```python
"""
Interest rate instrument definitions.

Covers the elementary instrument types used in IR static replication:
swaps, swaptions, caps/floors, and (optionally) basis swaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rade_sr.instruments.base import InstrumentSpec, TenorGrid


@dataclass
class IRSwap(InstrumentSpec):
    """Plain vanilla interest rate swap."""
    asset_class: str = "rates"
    payoff_type: str = "swap"
    factor_id: str = ""

    currency: str = ""
    fixed_rate: float = 0.0
    tenor: float = 0.0
    notional: float = 1.0
    pay_fixed: bool = True

    def to_pricer_params(self) -> Dict[str, Any]:
        """TODO(wire): Match to your IR swap pricer interface."""
        return {
            "currency": self.currency,
            "fixed_rate": self.fixed_rate,
            "tenor": self.tenor,
            "notional": self.notional,
            "pay_fixed": self.pay_fixed,
            "payoff_type": self.payoff_type,
        }

    def to_trade_id(self, cluster_id: str) -> str:
        direction = "pay" if self.pay_fixed else "rec"
        return f"ir_{self.currency}_swap_{direction}_{self.fixed_rate}_{self.tenor}_{cluster_id}"


@dataclass
class IRSwaption(InstrumentSpec):
    """European swaption (option to enter a swap)."""
    asset_class: str = "rates"
    payoff_type: str = "swaption"
    factor_id: str = ""

    currency: str = ""
    strike: float = 0.0
    option_expiry: float = 0.0
    swap_tenor: float = 0.0
    notional: float = 1.0
    is_payer: bool = True

    def to_pricer_params(self) -> Dict[str, Any]:
        """TODO(wire): Match to your swaption pricer.

        Typical inputs: strike (fixed rate), option expiry, underlying
        swap tenor, vol (normal or lognormal depending on convention),
        discount curve, projection curve.
        """
        return {
            "currency": self.currency,
            "strike": self.strike,
            "option_expiry": self.option_expiry,
            "swap_tenor": self.swap_tenor,
            "notional": self.notional,
            "is_payer": self.is_payer,
            "payoff_type": self.payoff_type,
        }

    def to_trade_id(self, cluster_id: str) -> str:
        side = "payer" if self.is_payer else "receiver"
        return (
            f"ir_{self.currency}_swaption_{side}_{self.strike}_"
            f"{self.option_expiry}x{self.swap_tenor}_{cluster_id}"
        )


@dataclass
class IRCapFloor(InstrumentSpec):
    """Interest rate cap or floor."""
    asset_class: str = "rates"
    payoff_type: str = "cap"
    factor_id: str = ""

    currency: str = ""
    strike: float = 0.0
    tenor: float = 0.0
    notional: float = 1.0
    is_cap: bool = True

    def to_pricer_params(self) -> Dict[str, Any]:
        """TODO(wire): Match to your cap/floor pricer interface."""
        return {
            "currency": self.currency,
            "strike": self.strike,
            "tenor": self.tenor,
            "notional": self.notional,
            "is_cap": self.is_cap,
            "payoff_type": "cap" if self.is_cap else "floor",
        }

    def to_trade_id(self, cluster_id: str) -> str:
        side = "cap" if self.is_cap else "floor"
        return f"ir_{self.currency}_{side}_{self.strike}_{self.tenor}_{cluster_id}"


def generate_ir_elementary_trades(
    currency: str,
    factor_id: str,
    cluster_id: str,
    tenor_grid: TenorGrid,
    strike_grid: Optional[List[float]] = None,
    payoff_types: Optional[List[str]] = None,
    notional: float = 1.0,
) -> List[InstrumentSpec]:
    """Generate the full grid of IR elementary instruments.

    TODO(wire): Adjust to match your existing IR trade generation logic.
    Typical approach: swaps at each liquid tenor, swaptions at each
    (option_expiry x swap_tenor) pair and strike, caps/floors at each
    tenor and strike.

    Parameters
    ----------
    currency : str
        Currency (e.g. ``"EUR"``, ``"USD"``).
    factor_id : str
        Risk factor identifier.
    cluster_id : str
        Originating cluster.
    tenor_grid : TenorGrid
        Grid of tenors from the rate curve.
    strike_grid : list[float] or None
        Fixed rate / strike values. If None, derive from par rates.
    payoff_types : list[str] or None
        Instrument types. Default: ``["swap"]``.
    notional : float
        Per-trade notional.
    """
    payoff_types = payoff_types or ["swap"]
    strike_grid = strike_grid or [0.0]
    trades: List[InstrumentSpec] = []

    for tenor in tenor_grid.values:
        if "swap" in payoff_types:
            for rate in strike_grid:
                trades.append(IRSwap(
                    factor_id=factor_id, currency=currency,
                    fixed_rate=rate, tenor=tenor, notional=notional,
                    pay_fixed=True,
                ))

        if "swaption" in payoff_types:
            for strike in strike_grid:
                for option_expiry in tenor_grid.values:
                    if option_expiry < tenor:
                        trades.append(IRSwaption(
                            factor_id=factor_id, currency=currency,
                            strike=strike, option_expiry=option_expiry,
                            swap_tenor=tenor, notional=notional,
                            is_payer=True,
                        ))

        if "cap" in payoff_types:
            for strike in strike_grid:
                trades.append(IRCapFloor(
                    factor_id=factor_id, currency=currency,
                    strike=strike, tenor=tenor, notional=notional,
                    is_cap=True,
                ))
        if "floor" in payoff_types:
            for strike in strike_grid:
                trades.append(IRCapFloor(
                    factor_id=factor_id, currency=currency,
                    strike=strike, tenor=tenor, notional=notional,
                    is_cap=False,
                ))

    return trades
```

---

## `market_data/__init__.py`

**Path:** `src/rade_sr/market_data/__init__.py`

```python
"""
Market data layer — loading, caching, and scenario generation.

Responsible for fetching raw market data from internal APIs or caches,
normalising it to house conventions, and building shock scenario sets.
"""
```

---

## `market_data/manager.py`

**Path:** `src/rade_sr/market_data/manager.py`

```python
"""
Market data manager — loads and caches market data per risk factor.

This is the single entry point for obtaining market data objects.
Asset-class-specific loading logic is dispatched internally; callers
see only ``load(factor_id) -> market_data_object``.

TODO(wire): This module wraps your existing market data loading logic.
The key integration points are:
1. ``from_config()`` — factory that reads API endpoints, cache paths,
   and asset-class-specific settings from your config.
2. ``load()`` — fetches and returns the market data object for a given
   factor_id. This is where your existing MarketDataManager / internal
   API calls go.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.rade_sr.core.exceptions import MarketDataError

logger = logging.getLogger(__name__)


class MarketDataManager:
    """Loads and caches market data for risk factors.

    Parameters
    ----------
    config : dict
        Market data configuration. Expected keys:

        - ``api_base_url`` (str): Internal API endpoint for market data.
        - ``cache_dir`` (str or None): Local cache directory. If provided,
          data is read from cache before hitting the API.
        - ``asset_class`` (str): Asset class for this manager instance
          (determines which loading logic to use).
        - Any asset-class-specific keys (e.g. ``vol_surface_type``,
          ``curve_hierarchy``, ``dividend_source``).

    TODO(wire): Replace the ``_load_*`` methods below with calls to your
    existing market data loading functions. The exact interface depends
    on your internal API client and data formats.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._cache: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, factor_config: Dict[str, Any]) -> MarketDataManager:
        """Factory: build a manager from per-factor pipeline config.

        Parameters
        ----------
        factor_config : dict
            The per-factor entry from ``PipelineConfig.asset_config``.
            Contains ``asset_class`` and any asset-specific keys.

        TODO(wire): Extract the relevant config keys for your internal
        API client construction here.
        """
        return cls(config=factor_config)

    def load(self, factor_id: str) -> Any:
        """Load market data for a single risk factor.

        Returns the fully loaded market data object (vol surface, curve,
        spot data, etc.). Caches on first load per factor_id within
        this manager instance.

        Parameters
        ----------
        factor_id : str
            Risk factor identifier (e.g. ``"EURUSD"``, ``"EUR_6M"``).

        Returns
        -------
        Any
            The market data object. Type depends on asset class:
            - FX: dict or object with spot, forward_curve, vol_surface
            - IR: dict or object with discount_curve, projection_curve, vol_cube
            - EQ: dict or object with spot, dividends, vol_surface
            - CR: dict or object with credit_spread_curve, recovery_rate

        Raises
        ------
        MarketDataError
            If loading fails (API error, missing data, validation failure).

        TODO(wire): Replace the body below with your actual loading logic.
        """
        if factor_id in self._cache:
            return self._cache[factor_id]

        asset_class = self._config.get("asset_class", "unknown")
        logger.info("Loading market data: factor=%s, asset_class=%s", factor_id, asset_class)

        try:
            if asset_class == "fx":
                data = self._load_fx(factor_id)
            elif asset_class == "rates":
                data = self._load_rates(factor_id)
            elif asset_class == "eq":
                data = self._load_equity(factor_id)
            elif asset_class == "cr":
                data = self._load_credit(factor_id)
            else:
                raise MarketDataError(
                    f"No market data loader for asset_class={asset_class!r}"
                )
        except MarketDataError:
            raise
        except Exception as exc:
            raise MarketDataError(
                f"Failed to load market data for {factor_id!r}: {exc}"
            ) from exc

        self._cache[factor_id] = data
        return data

    # ── Asset-class-specific loaders ──────────────────────────────────

    def _load_fx(self, factor_id: str) -> Any:
        """Load FX market data: spot, forward curve, vol surface.

        TODO(wire): Call your existing FX market data API here.
        Typical implementation:
            client = InternalAPIClient(self._config["api_base_url"])
            spot = client.get_fx_spot(factor_id)
            fwd_curve = client.get_fx_forwards(factor_id)
            vol_surface = client.get_fx_vol_surface(factor_id)
            return {"spot": spot, "forward_curve": fwd_curve, "vol_surface": vol_surface}
        """
        raise NotImplementedError("Wire to your FX market data loader")

    def _load_rates(self, factor_id: str) -> Any:
        """Load IR market data: discount curve, projection curve, vol cube.

        TODO(wire): Call your existing IR market data API here.
        Typical implementation:
            client = InternalAPIClient(self._config["api_base_url"])
            discount = client.get_discount_curve(factor_id)
            projection = client.get_projection_curve(factor_id)
            vol_cube = client.get_swaption_vol(factor_id)
            return {"discount": discount, "projection": projection, "vol_cube": vol_cube}
        """
        raise NotImplementedError("Wire to your IR market data loader")

    def _load_equity(self, factor_id: str) -> Any:
        """Load EQ market data: spot, dividends, vol surface.

        TODO(wire): Wire when EQ asset class is added.
        """
        raise NotImplementedError("Wire to your EQ market data loader")

    def _load_credit(self, factor_id: str) -> Any:
        """Load CR market data: credit spread curve, recovery rate.

        TODO(wire): Wire when CR asset class is added.
        """
        raise NotImplementedError("Wire to your CR market data loader")

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
```

---

## `market_data/scenarios.py`

**Path:** `src/rade_sr/market_data/scenarios.py`

```python
"""
Scenario / shock manager — builds scenario sets for risk factors.

Generates the shock scenarios that drive PnL computation. Each risk
factor gets a set of bumped/shifted market states that the pricer
evaluates the elementary trades against.

TODO(wire): This wraps your existing ShockManager / ScenarioManager.
The key integration point is ``build_scenarios()`` which produces
the scenario set for a given factor from its market data.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.rade_sr.core.exceptions import ScenarioError

logger = logging.getLogger(__name__)


class ScenarioManager:
    """Builds shock scenario sets for risk factors.

    Parameters
    ----------
    config : dict
        Scenario generation configuration. Expected keys:

        - ``n_scenarios`` (int): Number of shock scenarios to generate.
        - ``shock_type`` (str): Type of shock — ``"absolute"``, ``"relative"``,
          ``"historical"``, ``"monte_carlo"``.
        - ``seed`` (int or None): Random seed for reproducibility.
        - Asset-class-specific keys (e.g. ``vol_bump_sizes``,
          ``rate_shift_sizes``, ``spot_jump_sizes``).

    TODO(wire): Replace the ``_build_*`` methods with calls to your
    existing scenario generation logic.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    @classmethod
    def from_config(cls, factor_config: Dict[str, Any]) -> ScenarioManager:
        """Factory: build from per-factor pipeline config.

        TODO(wire): Extract scenario-relevant config keys from the
        factor_config dict passed through the pipeline.
        """
        return cls(config=factor_config)

    def build_scenarios(
        self,
        factor_id: str,
        market_data: Any,
        asset_class: Optional[str] = None,
    ) -> Any:
        """Build the shock scenario set for one risk factor.

        Parameters
        ----------
        factor_id : str
            Risk factor identifier.
        market_data : Any
            The loaded market data object (from MarketDataManager.load()).
            The scenario generator interrogates this to understand the
            base state that shocks are applied to.
        asset_class : str or None
            Asset class hint. If None, inferred from config.

        Returns
        -------
        Any
            The scenario set. Typical representations:
            - numpy array of shape ``(n_scenarios, n_dimensions)`` where
              dimensions depend on asset class (e.g. spot bumps, vol shifts,
              rate parallel shifts + key-rate bumps).
            - dict of named scenario arrays for multi-dimensional shocks.
            - Your existing ShockHistory or similar object.

        Raises
        ------
        ScenarioError
            If scenario generation fails.

        TODO(wire): Replace the body with your actual scenario generation.
        The exact API depends on your ShockManager. Typical pattern:
            shock_mgr = ShockManager(self._config)
            return shock_mgr.generate(factor_id, market_data)
        """
        ac = asset_class or self._config.get("asset_class", "unknown")
        logger.info(
            "Building scenarios: factor=%s, asset_class=%s, n=%s",
            factor_id, ac, self._config.get("n_scenarios", "?"),
        )

        try:
            if ac == "fx":
                return self._build_fx_scenarios(factor_id, market_data)
            elif ac == "rates":
                return self._build_ir_scenarios(factor_id, market_data)
            elif ac == "eq":
                return self._build_eq_scenarios(factor_id, market_data)
            elif ac == "cr":
                return self._build_cr_scenarios(factor_id, market_data)
            else:
                raise ScenarioError(f"No scenario builder for asset_class={ac!r}")
        except ScenarioError:
            raise
        except Exception as exc:
            raise ScenarioError(
                f"Failed to build scenarios for {factor_id!r}: {exc}"
            ) from exc

    # ── Asset-class-specific scenario builders ────────────────────────

    def _build_fx_scenarios(self, factor_id: str, market_data: Any) -> Any:
        """Build FX shock scenarios: spot bumps + vol surface shifts.

        TODO(wire): Call your existing FX scenario generation here.
        Typical implementation:
            from src.rade_sr.api.client import InternalAPIClient
            client = InternalAPIClient(self._config)
            shocks = client.get_fx_shocks(factor_id)
            return shocks  # or process into numpy arrays

        Or if generating locally:
            spot = market_data["spot"]
            n = self._config["n_scenarios"]
            seed = self._config.get("seed", 42)
            rng = np.random.default_rng(seed)
            spot_bumps = rng.normal(0, spot * 0.01, size=n)
            return {"spot_shocks": spot + spot_bumps}
        """
        raise NotImplementedError("Wire to your FX scenario generator")

    def _build_ir_scenarios(self, factor_id: str, market_data: Any) -> Any:
        """Build IR shock scenarios: parallel shifts + key-rate bumps + vol shifts.

        TODO(wire): Call your existing IR scenario generation here.
        IR scenarios are typically more complex than FX because the
        curve has multiple dimensions (pillar points). Common patterns:
        - Parallel shift: all pillars shifted by same amount
        - Key-rate: individual pillar bumps
        - PCA-based: scenarios from historical principal components
        """
        raise NotImplementedError("Wire to your IR scenario generator")

    def _build_eq_scenarios(self, factor_id: str, market_data: Any) -> Any:
        """Build EQ shock scenarios.

        TODO(wire): Wire when EQ asset class is added.
        """
        raise NotImplementedError("Wire to your EQ scenario generator")

    def _build_cr_scenarios(self, factor_id: str, market_data: Any) -> Any:
        """Build CR shock scenarios.

        TODO(wire): Wire when CR asset class is added.
        """
        raise NotImplementedError("Wire to your CR scenario generator")
```

---

## `orchestration/__init__.py`

**Path:** `src/rade_sr/orchestration/__init__.py`

```python
"""Orchestration — top-level pipeline wiring and execution."""
```

---

## `orchestration/orchestrator.py`

**Path:** `src/rade_sr/orchestration/orchestrator.py`

```python
"""
Orchestrator — wires all components and runs the preprocessing pipeline.

This is the outermost layer. It imports concrete implementations from
every other layer, constructs them, and hands them to the pipeline.

Usage
-----
::

    from src.rade_sr.orchestration.orchestrator import build_pipeline, run_preprocessing
    from src.rade_sr.config.loader import ConfigLoader

    config = ConfigLoader("/path/to/config").load_pipeline_config()
    pipeline = build_pipeline(cluster_manager, generator, encoder, pricer)
    jobs = run_preprocessing(pipeline, config)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from src.rade_sr.core.types import ReplicationJob
from src.rade_sr.replication.preprocessing.pipeline import (
    PipelineConfig,
    PreprocessingPipeline,
)
from src.rade_sr.replication.preprocessing.stages.cluster_resolver import ClusterResolver
from src.rade_sr.replication.preprocessing.stages.path_resolver import ConventionPathResolver
from src.rade_sr.replication.preprocessing.stages.trade_generator import (
    RiskFactorAwareTradeGenerator,
)
from src.rade_sr.replication.preprocessing.stages.pnl_engine import VectorisedPnLEngine

# Side-effect: registers all asset class builders with the registry
import src.rade_sr.replication.preprocessing.stages.asset_builders  # noqa: F401

logger = logging.getLogger(__name__)


def build_pipeline(
    cluster_manager: Any,
    generator: Any,
    encoder: Any,
    pricer: Any,
    max_workers: int = 4,
    hooks: Optional[List[Callable[[ReplicationJob], None]]] = None,
) -> PreprocessingPipeline:
    """Construct a fully wired PreprocessingPipeline.

    Parameters
    ----------
    cluster_manager : Any
        Your existing ClusterManager instance.

        TODO(wire): Type-hint with your actual class.
    generator : Any
        Your existing trade Generator instance.

        TODO(wire): Type-hint with your actual class.
    encoder : Any
        Your existing TradeEncoder instance.

        TODO(wire): Type-hint with your actual class.
    pricer : Any
        Your existing OptionPricer / PnlFactory instance.

        TODO(wire): Type-hint with your actual class.
    max_workers : int
        ProcessPoolExecutor concurrency.
    hooks : list[Callable] or None
        Post-assembly hooks for validation, logging, caching.

    Returns
    -------
    PreprocessingPipeline
        Ready to call ``.run(config)``.
    """
    return PreprocessingPipeline(
        cluster_resolver=ClusterResolver(cluster_manager),
        path_resolver=ConventionPathResolver(),
        trade_generator=RiskFactorAwareTradeGenerator(generator, encoder),
        pnl_engine=VectorisedPnLEngine(pricer, max_workers=max_workers),
        hooks=hooks,
    )


def run_preprocessing(
    pipeline: PreprocessingPipeline,
    config: PipelineConfig,
) -> List[ReplicationJob]:
    """Execute the preprocessing pipeline and return assembled jobs.

    Parameters
    ----------
    pipeline : PreprocessingPipeline
        Constructed via ``build_pipeline()``.
    config : PipelineConfig
        Loaded via ``ConfigLoader.load_pipeline_config()``.

    Returns
    -------
    list[ReplicationJob]
        One fully assembled job per cluster.
    """
    logger.info("Starting preprocessing pipeline")
    jobs = pipeline.run(config)
    logger.info(
        "Pipeline complete: %d jobs, %d total trades",
        len(jobs), sum(j.n_trades for j in jobs),
    )
    return jobs


# ── Example configuration (reference only) ────────────────────────────

EXAMPLE_CONFIG = PipelineConfig(
    cluster_config={
        "n_clusters": 20,
        "method": "user_defined",
        "cluster_key": "desk",
        "cluster_key_values": {
            "cluster_0": ["FX_FLOW"],
            "cluster_1": ["FX_EXOTIC"],
        },
    },
    path_config={
        "root": "/data/replication",
        "target_pnl_file": "target_pnl.parquet",
        "target_attr_file": "target_attributes.parquet",
        "elem_pnl_file": "elem_pnl.parquet",
        "elem_attr_file": "elem_attributes.parquet",
    },
    asset_config={
        "eurusd": {"asset_class": "fx", "pair": "EURUSD"},
        "gbpusd": {"asset_class": "fx", "pair": "GBPUSD"},
        "eur_6m": {"asset_class": "rates", "currency": "EUR"},
    },
    trade_config={
        "notional": 1_000_000,
        "expiry_grid": [0.25, 0.5, 1.0, 2.0],
        "payoff_types": ["call", "put", "forward"],
    },
    max_workers=8,
)
```

---

## `portfolio/__init__.py`

**Path:** `src/rade_sr/portfolio/__init__.py`

```python
"""Portfolio-level aggregation over ReplicationJob outputs."""
```

---

## `portfolio/manager.py`

**Path:** `src/rade_sr/portfolio/manager.py`

```python
"""
Portfolio manager — aggregates ReplicationJob outputs into portfolio-level views.

After the preprocessing pipeline produces a list of ReplicationJobs (one
per cluster), this manager combines them into portfolio-wide structures:
trade catalogues, aggregated PnL matrices, and job manifests.

This is the bridge between rade_sr (preprocessing) and rade_ml_pt (ML training).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.rade_sr.core.types import ReplicationJob

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Aggregates ReplicationJob outputs into portfolio-level structures.

    Parameters
    ----------
    jobs : list[ReplicationJob]
        Output from PreprocessingPipeline.run().
    """

    def __init__(self, jobs: List[ReplicationJob]) -> None:
        self._jobs = jobs
        self._job_map: Dict[str, ReplicationJob] = {j.cluster_id: j for j in jobs}

    @property
    def cluster_ids(self) -> List[str]:
        """Cluster IDs in pipeline order."""
        return [j.cluster_id for j in self._jobs]

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return len(self._jobs)

    @property
    def total_trades(self) -> int:
        """Total elementary trades across all clusters."""
        return sum(j.n_trades for j in self._jobs)

    def get_job(self, cluster_id: str) -> ReplicationJob:
        """Retrieve a specific cluster's job."""
        return self._job_map[cluster_id]

    def build_trade_catalogue(self) -> pd.DataFrame:
        """Build a portfolio-wide trade catalogue DataFrame.

        Each row is one elementary trade with its cluster membership,
        asset class, payoff type, factor, and parameters.

        Returns
        -------
        pd.DataFrame
            Columns: trade_id, cluster_id, asset_class, payoff_type,
            factor_id, notional, + flattened parameters.
        """
        rows: List[Dict[str, Any]] = []
        for job in self._jobs:
            for trade in job.cluster_elem_trades:
                row = {
                    "trade_id": trade.trade_id,
                    "cluster_id": trade.cluster_id,
                    "asset_class": trade.asset_class,
                    "payoff_type": trade.payoff_type,
                    "factor_id": trade.factor_id,
                    "notional": trade.notional,
                }
                row.update(trade.parameters)
                rows.append(row)
        return pd.DataFrame(rows)

    def build_portfolio_pnl_matrix(self) -> np.ndarray:
        """Build portfolio-wide PnL matrix by stacking cluster PnL matrices.

        Returns
        -------
        np.ndarray
            Shape ``(total_trades, n_scenarios)``.
            Row order matches ``build_trade_catalogue()`` row order.
        """
        matrices = [job.pnl_matrix() for job in self._jobs]
        return np.vstack(matrices)

    def build_cluster_mapping(self) -> Dict[str, List[str]]:
        """Build ``{cluster_id: [trade_ids]}`` mapping.

        This is the cluster_mapping format used by rade_ml_pt's
        EnsembleConfig.
        """
        return {
            job.cluster_id: job.trade_ids
            for job in self._jobs
        }

    def build_job_manifest(self) -> Dict[str, Any]:
        """Build a JSON-serialisable manifest for reproducibility.

        Contains cluster IDs, trade counts, factor counts, and
        total scenarios. Useful for audit trails and pipeline logging.
        """
        return {
            "n_clusters": self.n_clusters,
            "total_trades": self.total_trades,
            "cluster_summary": {
                job.cluster_id: {
                    "n_trades": job.n_trades,
                    "n_scenarios": job.n_scenarios,
                    "n_factors": len(job.factor_ids),
                    "factor_ids": job.factor_ids,
                }
                for job in self._jobs
            },
        }

    def to_ml_jobs(self) -> List[Dict[str, Any]]:
        """Convert all jobs to the dict format expected by rade_ml_pt.

        Returns
        -------
        list[dict]
            One dict per cluster in pipeline order, matching the
            ``job`` dict structure that ``build_dataset()`` expects.

        TODO(wire): Adjust the dict keys in ReplicationJob.to_job_dict()
        to match your exact rade_ml_pt build_dataset() interface.
        """
        return [job.to_job_dict() for job in self._jobs]
```

---

## `pricing/__init__.py`

**Path:** `src/rade_sr/pricing/__init__.py`

```python
"""
Pricing layer — PnL computation engines.

Provides batch pricing for elementary trades across shock scenarios.
The pricer is agnostic to the pipeline; it takes instrument parameters
and market data, returns PnL arrays.
"""
```

---

## `pricing/option_pricer.py`

**Path:** `src/rade_sr/pricing/option_pricer.py`

```python
"""
Option pricer — vectorised batch pricing across scenarios.

This is the computational core for PnL generation. The VectorisedPnLEngine
(in replication/preprocessing/stages/pnl_engine.py) delegates to this pricer
for each factor group.

The pricer must support:
1. ``extract_params(trade)`` — pull numeric parameters from an ElementaryTrade
2. ``price_batch(params, market_data, scenarios)`` — vectorised pricing

TODO(wire): This module wraps your existing PnlFactory / OptionPricer.
The key integration points are clearly marked below. Your existing pricer
likely already has batch pricing logic — this class provides a uniform
interface the pipeline can depend on.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.rade_sr.core.types import ElementaryTrade
from src.rade_sr.core.exceptions import PnLComputationError

logger = logging.getLogger(__name__)


class OptionPricer:
    """Batch pricer for elementary instruments across shock scenarios.

    Parameters
    ----------
    config : dict or None
        Pricer configuration. Keys depend on implementation:

        - ``model`` (str): Pricing model (``"black_scholes"``, ``"bachelier"``,
          ``"sabr"``, ``"local_vol"``).
        - ``day_count`` (str): Day count convention.
        - ``calendar`` (str): Business calendar.
        - Any model-specific parameters.

    TODO(wire): If your existing pricer is stateful (e.g. holds cached
    calibration data), pass it as a constructor argument or load it here.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    def extract_params(self, trade: ElementaryTrade) -> np.ndarray:
        """Extract a numeric parameter vector from a trade for batch pricing.

        The returned array is one row of the ``param_matrix`` passed to
        ``price_batch()``. Column order must be consistent across all
        trades of the same asset class.

        Parameters
        ----------
        trade : ElementaryTrade
            The elementary trade to extract parameters from.

        Returns
        -------
        np.ndarray
            1-D parameter vector. Typical FX example:
            ``[strike, expiry, is_call, notional]``

        TODO(wire): Map trade.parameters dict to a fixed-order numeric
        array. Match the column order your ``price_batch`` expects.

        Example for FX vanillas:
            return np.array([
                trade.parameters["strike"],
                trade.parameters["expiry"],
                1.0 if trade.parameters.get("is_call", True) else 0.0,
                trade.notional,
            ])
        """
        raise NotImplementedError("Wire to your parameter extraction logic")

    def price_batch(
        self,
        params: np.ndarray,
        market_data: Any,
        scenarios: Any,
    ) -> np.ndarray:
        """Price a batch of trades across all scenarios.

        This is the main vectorised pricing entry point. Called once per
        unique risk factor group by the VectorisedPnLEngine.

        Parameters
        ----------
        params : np.ndarray
            Parameter matrix, shape ``(n_trades, n_params)``.
            One row per trade; columns from ``extract_params()``.
        market_data : Any
            The risk factor's market data object (from MarketDataManager).
        scenarios : Any
            The risk factor's shock scenarios (from ScenarioManager).

        Returns
        -------
        np.ndarray
            PnL matrix, shape ``(n_trades, n_scenarios)``.
            ``result[i, j]`` = PnL of trade ``i`` under scenario ``j``.

        Raises
        ------
        PnLComputationError
            If pricing fails for this factor group.

        TODO(wire): Replace the body with your actual pricing logic.
        Typical pattern for FX options with Black-Scholes:

            spot = market_data["spot"]
            vol_surface = market_data["vol_surface"]
            scenarios_arr = scenarios["spot_shocks"]  # (n_scenarios,)

            n_trades = params.shape[0]
            n_scenarios = len(scenarios_arr)
            pnl = np.zeros((n_trades, n_scenarios), dtype=np.float64)

            for i in range(n_trades):
                strike = params[i, 0]
                expiry = params[i, 1]
                is_call = params[i, 2] > 0.5
                notional = params[i, 3]

                vol = vol_surface.get_vol(strike, expiry)
                base_price = black_scholes(spot, strike, expiry, vol, is_call)

                for j in range(n_scenarios):
                    shocked_spot = scenarios_arr[j]
                    shocked_price = black_scholes(shocked_spot, strike, expiry, vol, is_call)
                    pnl[i, j] = (shocked_price - base_price) * notional

            return pnl

        For vectorised (no inner loop) pricing, the above can be reshaped
        into (n_trades, 1) params broadcast against (1, n_scenarios) shocks.
        """
        raise NotImplementedError("Wire to your batch pricing logic")

    def price_single(
        self,
        trade: ElementaryTrade,
        market_data: Any,
        scenarios: Any,
    ) -> np.ndarray:
        """Convenience: price a single trade across scenarios.

        Returns shape ``(n_scenarios,)``.
        """
        params = self.extract_params(trade).reshape(1, -1)
        return self.price_batch(params, market_data, scenarios).squeeze(0)
```

---

## `replication/__init__.py`

**Path:** `src/rade_sr/replication/__init__.py`

```python
"""
Replication layer — core domain.

This is the reason the library exists. Contains the preprocessing
pipeline, risk factor registry, stage implementations, trade generation,
dimensionality reduction, clustering, and attribute encoding.
"""
```

---

## `replication/clustering/__init__.py`

**Path:** `src/rade_sr/replication/clustering/__init__.py`

```python
"""Clustering management for target trade partitioning."""
```

---

## `replication/clustering/manager.py`

**Path:** `src/rade_sr/replication/clustering/manager.py`

```python
"""
Cluster manager — partitions target trades into groups.

Supports both algorithmic clustering (k-means, hierarchical) and
user-defined clustering via explicit key-value mappings.

TODO(wire): This module wraps your existing clustering logic. The
ClusterResolver stage calls methods on this class via the injected
cluster_manager instance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of a clustering operation."""
    id: str
    trade_ids: List[str]
    metadata: Dict[str, Any]


class ClusterManager:
    """Manages trade clustering strategies.

    Parameters
    ----------
    trades : list[dict] or None
        Target trade records to cluster. Each dict should contain
        at least a ``trade_id`` and the attributes used for clustering.
    config : dict or None
        Clustering configuration.

    TODO(wire): Replace the clustering methods below with your existing
    implementation. If you use sklearn's KMeans or a custom clustering
    approach, wire it in ``get_clusters()``.
    """

    def __init__(
        self,
        trades: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._trades = trades or []
        self._config = config or {}

    def get_clusters(
        self,
        n_clusters: int,
        method: str = "kmeans",
        cluster_key: Optional[str] = None,
        cluster_key_values: Optional[Dict[str, List[Any]]] = None,
    ) -> List[ClusterResult]:
        """Partition trades into clusters.

        Parameters
        ----------
        n_clusters : int
            Number of clusters (for algorithmic methods).
        method : str
            Clustering method: ``"kmeans"``, ``"hierarchical"``,
            ``"user_defined"``.
        cluster_key : str or None
            Trade attribute key for user-defined clustering
            (e.g. ``"desk"``, ``"product_type"``).
        cluster_key_values : dict or None
            Explicit partition: ``{cluster_id: [attribute_values]}``.

        Returns
        -------
        list[ClusterResult]
            One result per cluster with trade IDs and metadata.

        TODO(wire): Replace with your actual clustering implementation.
        """
        if method == "user_defined" and cluster_key_values:
            return self._cluster_by_key(cluster_key, cluster_key_values)
        else:
            return self._cluster_algorithmic(n_clusters, method)

    def _cluster_by_key(
        self,
        key: Optional[str],
        values: Dict[str, List[Any]],
    ) -> List[ClusterResult]:
        """User-defined clustering by attribute key values.

        TODO(wire): Replace with your existing key-based partitioning.
        """
        raise NotImplementedError("Wire to your user-defined clustering logic")

    def _cluster_algorithmic(
        self,
        n_clusters: int,
        method: str,
    ) -> List[ClusterResult]:
        """Algorithmic clustering (k-means, hierarchical).

        TODO(wire): Replace with your existing algorithmic clustering.
        Typical pattern:
            from sklearn.cluster import KMeans
            features = self._extract_features()
            labels = KMeans(n_clusters=n_clusters).fit_predict(features)
            return self._labels_to_results(labels)
        """
        raise NotImplementedError(
            f"Wire to your {method} clustering implementation"
        )
```

---

## `replication/encoding/__init__.py`

**Path:** `src/rade_sr/replication/encoding/__init__.py`

```python
"""Trade attribute encoding and ID generation."""
```

---

## `replication/encoding/trade_encoder.py`

**Path:** `src/rade_sr/replication/encoding/trade_encoder.py`

```python
"""
Trade encoder — encodes trade attributes and generates unique trade IDs.

Provides deterministic trade ID encoding from instrument parameters and
trade attribute encoding (one-hot, multi-label, numeric) for the GNN.

TODO(wire): This wraps your existing TradeAttributeEncoder and trade ID
generation logic. If you already have encoding in rade_ml_pt, you can
import directly or duplicate the relevant parts here.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class TradeEncoder:
    """Encodes trade IDs and trade attributes.

    TODO(wire): Replace methods below with your existing encoding logic.
    """

    def encode_id(
        self,
        raw_trade: Any,
        cluster_id: str,
        factor_id: str,
    ) -> str:
        """Generate a deterministic, globally unique trade ID.

        Parameters
        ----------
        raw_trade : Any
            Raw trade object from the Generator. Must have attributes
            or dict keys for payoff_type, strike, expiry, etc.
        cluster_id : str
            Originating cluster.
        factor_id : str
            Risk factor this trade is associated with.

        Returns
        -------
        str
            Globally unique trade ID. Must be stable across runs for
            the same instrument parameters (for deduplication).

        TODO(wire): Replace with your existing ID encoding.
        Typical pattern:
            parts = [
                raw_trade.asset_class,
                factor_id,
                raw_trade.payoff_type,
                f"{raw_trade.strike:.4f}",
                f"{raw_trade.expiry:.2f}",
                cluster_id,
            ]
            return "_".join(parts)
        """
        raise NotImplementedError("Wire to your trade ID encoding logic")

    def encode_attributes(
        self,
        trade_attribs: Dict[str, List[Any]],
        encoding_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Encode trade attributes for the GNN feature matrix.

        Parameters
        ----------
        trade_attribs : dict
            ``{attribute_name: [values_per_trade]}``.
        encoding_config : dict or None
            Per-attribute encoding specification (one_hot, numeric, etc.).

        Returns
        -------
        dict
            ``{feature_name: np.ndarray}`` with encoded features.

        TODO(wire): Replace with your existing TradeAttributeEncoder.
        If you use the encoder from rade_ml_pt, import directly:
            from src.rade_ml_pt.utilities.attribute_encoder import TradeAttributeEncoder
        """
        raise NotImplementedError("Wire to your attribute encoding logic")


# Optional typing import for the encode_attributes signature
from typing import Optional  # noqa: E402
```

---

## `replication/preprocessing/__init__.py`

**Path:** `src/rade_sr/replication/preprocessing/__init__.py`

```python
"""
Preprocessing pipeline — two-phase orchestration of static replication stages.
"""
```

---

## `replication/preprocessing/pipeline.py`

**Path:** `src/rade_sr/replication/preprocessing/pipeline.py`

```python
"""
Two-phase preprocessing pipeline. See DESIGN.md for full rationale.

Phase 1 — per cluster, embarrassingly parallel:
    Stage 1: PathResolver         → ClusterPaths
    Stage 2: RiskFactorRegistry   → dict[factor_id, RiskFactor]
    Stage 3: TradeGenerator       → list[ElementaryTrade]
    Output:  IntermediateClusterData

Phase 2 — portfolio-wide then distributed:
    Stage 4: PnLEngine.compute_batch() → dict[trade_id, ElementaryTradePnL]
    Stage 5: distribute + assemble     → ReplicationJob per cluster + hooks
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from src.rade_sr.core.types import (
    ElementaryTrade,
    IntermediateClusterData,
    ReplicationJob,
    RiskFactor,
    TradeGenerationContext,
)
from src.rade_sr.core.protocols import (
    ClusterResolver,
    PathResolver,
    PnLEngine,
    TradeGenerator,
)
from src.rade_sr.core.exceptions import PipelineError
from src.rade_sr.replication.preprocessing.registry import RiskFactorRegistry

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline.

    Parameters
    ----------
    cluster_config : dict
        Passed to ``ClusterResolver.resolve()``. Keys depend on your
        clustering strategy (n_clusters, method, cluster_key, etc.).
    path_config : dict
        Root directory + filename conventions for cluster artifacts.
    asset_config : dict
        ``{factor_id: {asset_class, pair/currency, ...}}``.
        Defines which risk factors to build and their asset-class-specific
        configuration.
    trade_config : dict
        Trade generation parameters: notional, expiry grid, payoff types, etc.
    max_workers : int
        ProcessPoolExecutor concurrency for Phase 1 and Phase 2.
    fail_fast : bool
        If True, abort the pipeline on the first cluster failure.
        If False, log failures and continue with remaining clusters.
    """
    cluster_config: Dict[str, Any] = field(default_factory=dict)
    path_config: Dict[str, Any] = field(default_factory=dict)
    asset_config: Dict[str, Any] = field(default_factory=dict)
    trade_config: Dict[str, Any] = field(default_factory=dict)
    max_workers: int = 4
    fail_fast: bool = True


class PreprocessingPipeline:
    """Two-phase preprocessing pipeline for static replication.

    The pipeline depends ONLY on Protocol types from core/protocols.py.
    It has zero imports from instruments/, market_data/, or concrete stage
    implementations. Any stage can be swapped by passing a different object
    at construction time.

    Parameters
    ----------
    cluster_resolver : ClusterResolver
        Resolves config → list of cluster IDs.
    path_resolver : PathResolver
        Resolves cluster_id → ClusterPaths.
    trade_generator : TradeGenerator
        Generates elementary trades from assembled risk factors.
    pnl_engine : PnLEngine
        Computes portfolio-wide PnL.
    registry : type[RiskFactorRegistry] or None
        Risk factor registry class (default: RiskFactorRegistry).
    hooks : list[Callable] or None
        Post-assembly hooks. Each receives a ReplicationJob.
        Use for validation, logging, caching, database writes.
    """

    def __init__(
        self,
        cluster_resolver: ClusterResolver,
        path_resolver: PathResolver,
        trade_generator: TradeGenerator,
        pnl_engine: PnLEngine,
        registry: Optional[Type[RiskFactorRegistry]] = None,
        hooks: Optional[List[Callable[[ReplicationJob], None]]] = None,
    ) -> None:
        self._cluster_resolver = cluster_resolver
        self._path_resolver = path_resolver
        self._trade_generator = trade_generator
        self._pnl_engine = pnl_engine
        self._registry = registry or RiskFactorRegistry
        self._hooks = hooks or []

    # ── Public entry point ────────────────────────────────────────────

    def run(self, config: PipelineConfig) -> List[ReplicationJob]:
        """Execute the full two-phase preprocessing pipeline.

        Parameters
        ----------
        config : PipelineConfig
            Full pipeline configuration.

        Returns
        -------
        list[ReplicationJob]
            One fully assembled job per cluster, in cluster_ids order.
        """
        cluster_ids = self._cluster_resolver.resolve(config.cluster_config)
        logger.info("Resolved %d clusters: %s", len(cluster_ids), cluster_ids)

        intermediates = self._phase1(cluster_ids, config)
        logger.info(
            "Phase 1 complete: %d intermediates, %d total trades",
            len(intermediates),
            sum(i.n_trades for i in intermediates),
        )

        jobs = self._phase2(intermediates, config)
        logger.info("Phase 2 complete: %d jobs assembled.", len(jobs))

        return jobs

    # ── Phase 1: per-cluster, parallel ────────────────────────────────

    def _phase1(
        self,
        cluster_ids: List[str],
        config: PipelineConfig,
    ) -> List[IntermediateClusterData]:
        """Build IntermediateClusterData for each cluster.

        Runs each cluster independently in a ProcessPoolExecutor.
        Results are sorted back into the original cluster_ids order.
        """
        results: List[IntermediateClusterData] = []

        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(
                    _build_intermediate, cid, config,
                ): cid
                for cid in cluster_ids
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("Phase 1 cluster %s failed: %s", cid, exc)
                    if config.fail_fast:
                        raise PipelineError(
                            f"Phase 1 aborted at cluster {cid!r}"
                        ) from exc

        order = {cid: i for i, cid in enumerate(cluster_ids)}
        return sorted(results, key=lambda d: order[d.cluster_id])

    # ── Phase 2: portfolio-wide PnL + distribution ────────────────────

    def _phase2(
        self,
        intermediates: List[IntermediateClusterData],
        config: PipelineConfig,
    ) -> List[ReplicationJob]:
        """Compute portfolio-wide PnL and distribute back to clusters."""
        all_trades: List[ElementaryTrade] = []
        all_risk_factors: Dict[str, RiskFactor] = {}

        for intermed in intermediates:
            all_trades.extend(intermed.cluster_elem_trades)
            for fid, rf in intermed.cluster_assets.items():
                if fid not in all_risk_factors:
                    all_risk_factors[fid] = rf

        logger.info(
            "Phase 2: pricing %d trades across %d unique risk factors",
            len(all_trades), len(all_risk_factors),
        )

        all_pnls = self._pnl_engine.compute_batch(all_trades, all_risk_factors)

        jobs: List[ReplicationJob] = []
        for intermed in intermediates:
            cluster_pnls = {
                t.trade_id: all_pnls[t.trade_id]
                for t in intermed.cluster_elem_trades
                if t.trade_id in all_pnls
            }
            job = ReplicationJob(
                cluster_id=intermed.cluster_id,
                cluster_info=intermed.cluster_info,
                cluster_assets=intermed.cluster_assets,
                cluster_elem_trades=intermed.cluster_elem_trades,
                elem_trade_pnls=cluster_pnls,
            )
            self._run_hooks(job)
            jobs.append(job)

        return jobs

    def _run_hooks(self, job: ReplicationJob) -> None:
        """Execute post-assembly hooks."""
        for hook in self._hooks:
            try:
                hook(job)
            except Exception as exc:
                logger.warning(
                    "Hook %s raised for cluster %s: %s",
                    getattr(hook, "__name__", repr(hook)),
                    job.cluster_id,
                    exc,
                )


# ── Module-level pure function for Phase 1 worker processes ──────────

def _build_intermediate(
    cluster_id: str,
    config: PipelineConfig,
) -> IntermediateClusterData:
    """Build IntermediateClusterData for one cluster.

    IMPORTANT: This is a module-level pure function, NOT a method.
    ProcessPoolExecutor pickles functions and arguments to ship them
    to worker processes. Any closure over non-serialisable objects
    (database sessions, open file handles, locks) will cause PicklingError.

    All dependencies are instantiated locally inside this function.
    This also makes it independently testable without a process pool.

    TODO(wire): The local imports below instantiate your concrete stage
    implementations. Adjust them if your classes live elsewhere or have
    different constructor signatures.
    """
    from src.rade_sr.replication.preprocessing.stages.path_resolver import (
        ConventionPathResolver,
    )
    from src.rade_sr.replication.preprocessing.registry import RiskFactorRegistry
    from src.rade_sr.replication.preprocessing.stages.trade_generator import (
        RiskFactorAwareTradeGenerator,
    )
    from src.rade_sr.replication.trades.generator import Generator
    from src.rade_sr.replication.encoding.trade_encoder import TradeEncoder

    # Side-effect import: registers all asset class builders
    import src.rade_sr.replication.preprocessing.stages.asset_builders  # noqa: F401

    # Stage 1: resolve paths
    paths = ConventionPathResolver().resolve(cluster_id, config.path_config)

    # Stage 2: build risk factors via registry dispatch
    cluster_assets: Dict[str, RiskFactor] = {}
    for factor_id, factor_cfg in config.asset_config.items():
        builder = RiskFactorRegistry.get(factor_cfg["asset_class"])
        cluster_assets[factor_id] = builder.build(
            factor_id, factor_cfg, cluster_id,
        )

    # Stage 3: generate elementary trades
    context = TradeGenerationContext(
        cluster_id=cluster_id,
        risk_factors=cluster_assets,
        trade_config=config.trade_config,
    )
    elem_trades = RiskFactorAwareTradeGenerator(
        Generator(), TradeEncoder(),
    ).generate(context)

    return IntermediateClusterData(
        cluster_id=cluster_id,
        cluster_info=paths,
        cluster_assets=cluster_assets,
        cluster_elem_trades=elem_trades,
    )
```

---

## `replication/preprocessing/registry.py`

**Path:** `src/rade_sr/replication/preprocessing/registry.py`

```python
"""
Risk factor registry — decorator-based asset class dispatch.

Maps ``asset_class: str → RiskFactorBuilder`` instance. Registration
happens via ``@RiskFactorRegistry.register`` at import time.

Importing ``stages/asset_builders.py`` is the registration act.
The pipeline never needs to know which asset classes exist — it calls
``registry.get(asset_class)`` and dispatches blindly.

Adding a new asset class = one new decorated class in asset_builders.py.
Nothing else changes.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Type

from src.rade_sr.core.protocols import RiskFactorBuilder
from src.rade_sr.core.exceptions import UnknownAssetClassError

logger = logging.getLogger(__name__)


class RiskFactorRegistry:
    """Maps asset_class string to RiskFactorBuilder instance.

    Usage
    -----
    ::

        @RiskFactorRegistry.register
        class FXRiskFactorBuilder:
            asset_class = "fx"
            def build(self, factor_id, factor_config, cluster_id) -> RiskFactor: ...

    Then elsewhere::

        builder = RiskFactorRegistry.get("fx")
        risk_factor = builder.build("EURUSD", config, "cluster_0")
    """

    _registry: Dict[str, RiskFactorBuilder] = {}

    @classmethod
    def register(cls, builder_cls: Type) -> Type:
        """Class decorator that registers a RiskFactorBuilder implementation.

        The class must define ``asset_class`` as a class-level string attribute.
        An instance is created and stored in the registry.

        Parameters
        ----------
        builder_cls : type
            The builder class to register.

        Returns
        -------
        type
            The same class (unmodified), allowing normal instantiation.

        Raises
        ------
        TypeError
            If the class doesn't satisfy the RiskFactorBuilder protocol.
        AttributeError
            If the class doesn't define ``asset_class``.
        """
        instance = builder_cls()
        if not isinstance(instance, RiskFactorBuilder):
            raise TypeError(
                f"{builder_cls.__name__} does not satisfy RiskFactorBuilder protocol."
            )
        key = getattr(builder_cls, "asset_class", None)
        if not key:
            raise AttributeError(
                f"{builder_cls.__name__} must define `asset_class` as a class attribute."
            )
        cls._registry[key] = instance
        logger.debug("Registered RiskFactorBuilder: %s -> %s", key, builder_cls.__name__)
        return builder_cls

    @classmethod
    def get(cls, asset_class: str) -> RiskFactorBuilder:
        """Look up the builder for a given asset class.

        Raises
        ------
        UnknownAssetClassError
            If no builder is registered for the requested asset class.
        """
        try:
            return cls._registry[asset_class]
        except KeyError:
            raise UnknownAssetClassError(
                f"No builder registered for asset_class={asset_class!r}. "
                f"Registered: {list(cls._registry)}"
            )

    @classmethod
    def registered(cls) -> List[str]:
        """List all registered asset class keys."""
        return list(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (primarily for testing)."""
        cls._registry.clear()
```

---

## `replication/preprocessing/stages/__init__.py`

**Path:** `src/rade_sr/replication/preprocessing/stages/__init__.py`

```python
"""
Pipeline stage implementations.

Each stage wraps existing business logic behind a Protocol-compatible
interface. The pipeline depends on Protocols, not on these concrete classes.
"""
```

---

## `replication/preprocessing/stages/asset_builders.py`

**Path:** `src/rade_sr/replication/preprocessing/stages/asset_builders.py`

```python
"""
Asset-class-specific RiskFactorBuilder implementations.

IMPORTANT: Importing this module is the registration act. The pipeline
never imports this directly — the orchestrator does it as a side effect:

    import src.rade_sr.replication.preprocessing.stages.asset_builders

Each decorated class is automatically registered with the RiskFactorRegistry.
Adding a new asset class = one new decorated class here. The pipeline itself
does not change.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from src.rade_sr.core.types import RiskFactor
from src.rade_sr.replication.preprocessing.registry import RiskFactorRegistry

logger = logging.getLogger(__name__)


@RiskFactorRegistry.register
class FXRiskFactorBuilder:
    """Builds an FX risk factor: spot + vol surface + forward curve + scenarios.

    TODO(wire): Replace the body of ``build()`` with calls to your existing
    MarketDataManager and ScenarioManager (or ShockManager) for FX data.
    """
    asset_class = "fx"

    def build(
        self,
        factor_id: str,
        factor_config: Dict[str, Any],
        cluster_id: str,
    ) -> RiskFactor:
        """Load FX market data + scenarios for one risk factor.

        Parameters
        ----------
        factor_id : str
            Currency pair identifier (e.g. ``"EURUSD"``).
        factor_config : dict
            Per-factor config. Expected FX-specific keys:
            - ``pair`` (str): Currency pair.
            - ``api_base_url`` (str): Market data API endpoint.
            - ``n_scenarios`` (int): Number of shock scenarios.
            - Any additional keys your loaders need.
        cluster_id : str
            Originating cluster (for metadata tagging).

        TODO(wire): Typical implementation:

            from src.rade_sr.market_data.manager import MarketDataManager
            from src.rade_sr.market_data.scenarios import ScenarioManager

            mdm = MarketDataManager.from_config(factor_config)
            market_data = mdm.load(factor_id)

            scenarios = ScenarioManager.from_config(factor_config).build_scenarios(
                factor_id, market_data, asset_class="fx"
            )

            return RiskFactor(
                factor_id=factor_id,
                asset_class="fx",
                market_data=market_data,
                scenarios=scenarios,
                metadata={"pair": factor_config.get("pair"), "cluster": cluster_id},
            )
        """
        from src.rade_sr.market_data.manager import MarketDataManager
        from src.rade_sr.market_data.scenarios import ScenarioManager

        mdm = MarketDataManager.from_config(factor_config)
        market_data = mdm.load(factor_id)

        scenarios = ScenarioManager.from_config(factor_config).build_scenarios(
            factor_id, market_data, asset_class="fx",
        )

        return RiskFactor(
            factor_id=factor_id,
            asset_class="fx",
            market_data=market_data,
            scenarios=scenarios,
            metadata={"pair": factor_config.get("pair"), "cluster": cluster_id},
        )


@RiskFactorRegistry.register
class RatesRiskFactorBuilder:
    """Builds an IR risk factor: yield curves + swaption vol + scenarios.

    TODO(wire): Replace with calls to your IR market data loading.
    """
    asset_class = "rates"

    def build(
        self,
        factor_id: str,
        factor_config: Dict[str, Any],
        cluster_id: str,
    ) -> RiskFactor:
        """Load IR market data + scenarios for one risk factor.

        Parameters
        ----------
        factor_id : str
            IR factor identifier (e.g. ``"EUR_6M"``, ``"USD_OIS"``).
        factor_config : dict
            Per-factor config. Expected IR-specific keys:
            - ``currency`` (str): Currency.
            - ``curve_type`` (str): ``"discount"``, ``"projection"``, etc.

        TODO(wire): Same pattern as FX but with IR-specific loaders.
        """
        from src.rade_sr.market_data.manager import MarketDataManager
        from src.rade_sr.market_data.scenarios import ScenarioManager

        mdm = MarketDataManager.from_config(factor_config)
        market_data = mdm.load(factor_id)

        scenarios = ScenarioManager.from_config(factor_config).build_scenarios(
            factor_id, market_data, asset_class="rates",
        )

        return RiskFactor(
            factor_id=factor_id,
            asset_class="rates",
            market_data=market_data,
            scenarios=scenarios,
            metadata={"currency": factor_config.get("currency"), "cluster": cluster_id},
        )


# ── Templates for new asset classes ──────────────────────────────────────

# @RiskFactorRegistry.register
# class EqRiskFactorBuilder:
#     """Builds an EQ risk factor: spot + dividends + vol surface + scenarios."""
#     asset_class = "eq"
#
#     def build(self, factor_id, factor_config, cluster_id) -> RiskFactor:
#         # TODO(wire): Implement for equity.
#         ...

# @RiskFactorRegistry.register
# class CreditRiskFactorBuilder:
#     """Builds a CR risk factor: credit spread curve + recovery + scenarios."""
#     asset_class = "cr"
#
#     def build(self, factor_id, factor_config, cluster_id) -> RiskFactor:
#         # TODO(wire): Implement for credit.
#         ...
```

---

## `replication/preprocessing/stages/cluster_resolver.py`

**Path:** `src/rade_sr/replication/preprocessing/stages/cluster_resolver.py`

```python
"""
Cluster resolver stage — wraps existing clustering logic.

Translates user configuration into a flat list of cluster ID strings.
The pipeline only sees string IDs; clustering strategy details are
encapsulated here.

TODO(wire): Replace the internal call with your existing ClusterManager API.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ClusterResolver:
    """Wraps the existing cluster manager to satisfy the Protocol.

    Parameters
    ----------
    cluster_manager : Any
        Your existing ClusterManager instance. Injected at construction time
        by the orchestrator.

        TODO(wire): Type-hint this with your actual ClusterManager class
        once you wire it up.
    """

    def __init__(self, cluster_manager: Any) -> None:
        self._cm = cluster_manager

    def resolve(self, config: Dict[str, Any]) -> List[str]:
        """Resolve clustering configuration into a list of cluster IDs.

        Parameters
        ----------
        config : dict
            Clustering configuration. Typical keys:

            - ``n_clusters`` (int): Number of clusters.
            - ``method`` (str): Clustering method (e.g. ``"kmeans"``, ``"user_defined"``).
            - ``cluster_key`` (str): Attribute key used for partitioning
              (e.g. ``"desk"``, ``"product_type"``, ``"ccy"``).
            - ``cluster_key_values`` (dict): Pre-defined cluster partitions
              ``{cluster_id: [list of key values]}``.

        Returns
        -------
        list[str]
            Ordered list of cluster ID strings.

        TODO(wire): Replace the body with your actual clustering call.
        Typical pattern:

            clusters = self._cm.get_clusters(
                n_clusters=config["n_clusters"],
                method=config.get("method", "kmeans"),
            )
            return [str(c.id) for c in clusters]

        Or for user-defined clusters:

            return list(config["cluster_key_values"].keys())
        """
        raise NotImplementedError(
            "Wire to your ClusterManager.get_clusters() or equivalent"
        )
```

---

## `replication/preprocessing/stages/path_resolver.py`

**Path:** `src/rade_sr/replication/preprocessing/stages/path_resolver.py`

```python
"""
Path resolver stage — derives artifact paths per cluster.

Translates a root directory + naming conventions into the four canonical
file paths (target PnL, target attributes, elementary PnL, elementary
attributes) for each cluster.

Subclass or replace to support different path layouts (e.g. database-backed,
cloud storage, date-partitioned directories).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.rade_sr.core.types import ClusterPaths

logger = logging.getLogger(__name__)


class ConventionPathResolver:
    """Derives artifact paths from a root directory + naming convention.

    The convention is: ``{root}/{cluster_id}/{filename}`` for each of the
    four artifact types. Filenames come from ``path_config``.

    This resolver is stateless and can be shared across clusters.
    """

    def resolve(self, cluster_id: str, path_config: Dict[str, Any]) -> ClusterPaths:
        """Resolve the four artifact paths for one cluster.

        Parameters
        ----------
        cluster_id : str
            Cluster identifier (used as subdirectory name).
        path_config : dict
            Path configuration. Required keys:

            - ``root`` (str): Root directory for all cluster data.
            - ``target_pnl_file`` (str): Filename for target PnL
              (e.g. ``"target_pnl.parquet"``).
            - ``target_attr_file`` (str): Filename for target attributes.
            - ``elem_pnl_file`` (str): Filename for elementary PnL.
            - ``elem_attr_file`` (str): Filename for elementary attributes.

        Returns
        -------
        ClusterPaths
            Frozen dataclass with the four resolved paths.

        TODO(wire): If your path convention is different (e.g. paths include
        dates, versions, or are not cluster-subdirectories), override this
        method or write a new PathResolver implementation.
        """
        root = Path(path_config["root"])
        cluster_dir = root / cluster_id

        return ClusterPaths(
            target_pnl=cluster_dir / path_config["target_pnl_file"],
            target_attributes=cluster_dir / path_config["target_attr_file"],
            elem_pnl=cluster_dir / path_config["elem_pnl_file"],
            elem_attributes=cluster_dir / path_config["elem_attr_file"],
        )
```

---

## `replication/preprocessing/stages/pnl_engine.py`

**Path:** `src/rade_sr/replication/preprocessing/stages/pnl_engine.py`

```python
"""
PnL engine stage — portfolio-wide vectorised PnL computation.

Groups all elementary trades by factor_id, makes one vectorised pricer
call per unique factor group, and returns a flat dict of per-trade PnL
vectors. Factor groups are computed in parallel.

This runs in Phase 2 (portfolio-wide) to avoid redundant pricing of
shared risk factors across clusters.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np

from src.rade_sr.core.types import ElementaryTrade, ElementaryTradePnL, RiskFactor
from src.rade_sr.core.exceptions import PnLComputationError

logger = logging.getLogger(__name__)


class VectorisedPnLEngine:
    """Portfolio-wide PnL computation engine.

    Groups trades by factor_id, then computes PnL for each group in
    a single vectorised pricer call. Factor groups can be processed
    in parallel via ProcessPoolExecutor.

    Parameters
    ----------
    pricer : Any
        Your existing OptionPricer / PnlFactory instance.

        TODO(wire): Type-hint with your actual pricer class. The pricer
        must support:
        - ``extract_params(trade) -> np.ndarray``
        - ``price_batch(params, market_data, scenarios) -> np.ndarray``
          returning shape ``(n_trades, n_scenarios)``.
    max_workers : int
        Number of parallel workers for factor group processing.
    """

    def __init__(self, pricer: Any, max_workers: int = 4) -> None:
        self._pricer = pricer
        self._max_workers = max_workers

    def compute_batch(
        self,
        trades: List[ElementaryTrade],
        risk_factors: Dict[str, RiskFactor],
    ) -> Dict[str, ElementaryTradePnL]:
        """Compute PnL for all trades, grouped by risk factor.

        Parameters
        ----------
        trades : list[ElementaryTrade]
            Portfolio-wide list of all elementary trades (all clusters).
        risk_factors : dict[str, RiskFactor]
            Deduplicated risk factors keyed by factor_id.

        Returns
        -------
        dict[str, ElementaryTradePnL]
            Mapping from trade_id to its PnL vector.
        """
        groups = self._group_by_factor(trades)
        results: Dict[str, ElementaryTradePnL] = {}

        logger.info(
            "PnL engine: %d trades across %d factor groups, %d workers",
            len(trades), len(groups), self._max_workers,
        )

        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_factor = {
                executor.submit(
                    self._price_group,
                    factor_id,
                    group_trades,
                    risk_factors[factor_id],
                ): factor_id
                for factor_id, group_trades in groups.items()
                if factor_id in risk_factors
            }
            for future in as_completed(future_to_factor):
                factor_id = future_to_factor[future]
                try:
                    results.update(future.result())
                    logger.debug("Priced factor group: %s", factor_id)
                except Exception as exc:
                    raise PnLComputationError(
                        f"PnL computation failed for factor {factor_id!r}"
                    ) from exc

        logger.info("PnL engine: computed %d trade PnLs", len(results))
        return results

    @staticmethod
    def _group_by_factor(
        trades: List[ElementaryTrade],
    ) -> Dict[str, List[ElementaryTrade]]:
        """Group trades by their risk factor ID."""
        groups: Dict[str, List[ElementaryTrade]] = defaultdict(list)
        for t in trades:
            groups[t.factor_id].append(t)
        return dict(groups)

    def _price_group(
        self,
        factor_id: str,
        trades: List[ElementaryTrade],
        risk_factor: RiskFactor,
    ) -> Dict[str, ElementaryTradePnL]:
        """Price all trades for one risk factor group.

        This method is submitted to ProcessPoolExecutor and must be
        safe across process boundaries.

        TODO(wire): The ``self._pricer.extract_params(t)`` and
        ``self._pricer.price_batch(...)`` calls are the key integration
        points. Wire them to your existing PnlFactory or OptionPricer.

        Parameters
        ----------
        factor_id : str
            The risk factor being priced.
        trades : list[ElementaryTrade]
            All trades referencing this factor.
        risk_factor : RiskFactor
            The fully loaded risk factor with market data and scenarios.

        Returns
        -------
        dict[str, ElementaryTradePnL]
            Per-trade PnL vectors for this factor group.
        """
        param_matrix = np.array([
            self._pricer.extract_params(t) for t in trades
        ])

        pnl_matrix = self._pricer.price_batch(
            params=param_matrix,
            market_data=risk_factor.market_data,
            scenarios=risk_factor.scenarios,
        )

        return {
            trade.trade_id: ElementaryTradePnL(
                trade_id=trade.trade_id,
                factor_id=factor_id,
                cluster_id=trade.cluster_id,
                pnl_vector=pnl_matrix[i],
            )
            for i, trade in enumerate(trades)
        }
```

---

## `replication/preprocessing/stages/trade_generator.py`

**Path:** `src/rade_sr/replication/preprocessing/stages/trade_generator.py`

```python
"""
Trade generator stage — generates elementary trades from assembled risk factors.

The generator interrogates each risk factor's market data to produce
contextually appropriate instruments (strike grids from vol surfaces,
tenor grids from rate curves, payoff types for the asset class).

Deduplication ensures the same elementary trade (e.g. a 3M EURUSD 1.10
call) is not generated twice even if it appears in multiple clusters.

TODO(wire): The internal ``_generate_for_factor()`` method is where your
existing trade generation logic should be called.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from src.rade_sr.core.types import (
    ElementaryTrade,
    RiskFactor,
    TradeGenerationContext,
)
from src.rade_sr.core.exceptions import TradeGenerationError

logger = logging.getLogger(__name__)


class RiskFactorAwareTradeGenerator:
    """Generates elementary trades by interrogating each risk factor's market data.

    Parameters
    ----------
    generator : Any
        Your existing Generator instance (from replication/trades/generator.py).
        Handles the actual instrument creation logic per asset class.

        TODO(wire): Type-hint with your actual Generator class.
    encoder : Any
        Your existing TradeEncoder instance (from replication/encoding/trade_encoder.py).
        Handles trade ID encoding.

        TODO(wire): Type-hint with your actual TradeEncoder class.
    """

    def __init__(self, generator: Any, encoder: Any) -> None:
        self._generator = generator
        self._encoder = encoder

    def generate(self, context: TradeGenerationContext) -> List[ElementaryTrade]:
        """Generate and deduplicate elementary trades for one cluster.

        Iterates over all risk factors in the context, generates trades
        for each, then deduplicates by trade_id.

        Parameters
        ----------
        context : TradeGenerationContext
            Contains cluster_id, assembled risk factors, and trade config.

        Returns
        -------
        list[ElementaryTrade]
            Deduplicated list of elementary trades for this cluster.
        """
        trades: List[ElementaryTrade] = []
        for factor_id, risk_factor in context.risk_factors.items():
            try:
                factor_trades = self._generate_for_factor(
                    factor_id, risk_factor, context,
                )
                trades.extend(factor_trades)
                logger.debug(
                    "Generated %d trades for factor %s (cluster %s)",
                    len(factor_trades), factor_id, context.cluster_id,
                )
            except Exception as exc:
                raise TradeGenerationError(
                    f"Trade generation failed for factor {factor_id!r} "
                    f"in cluster {context.cluster_id!r}: {exc}"
                ) from exc

        deduplicated = self._deduplicate(trades)
        logger.info(
            "Cluster %s: %d trades generated, %d after dedup",
            context.cluster_id, len(trades), len(deduplicated),
        )
        return deduplicated

    def _generate_for_factor(
        self,
        factor_id: str,
        risk_factor: RiskFactor,
        context: TradeGenerationContext,
    ) -> List[ElementaryTrade]:
        """Generate trades for one risk factor within a cluster.

        TODO(wire): Replace the body with your existing trade generation call.
        Typical pattern:

            raw_trades = self._generator.generate_for_factor(
                factor_id=factor_id,
                asset_class=risk_factor.asset_class,
                market_data=risk_factor.market_data,
                scenarios=risk_factor.scenarios,
                params=context.trade_config,
            )

            return [
                ElementaryTrade(
                    trade_id=self._encoder.encode_id(t, context.cluster_id, factor_id),
                    asset_class=risk_factor.asset_class,
                    payoff_type=t.payoff_type,
                    factor_id=factor_id,
                    parameters=t.to_dict(),
                    notional=context.trade_config.get("notional", 1.0),
                    cluster_id=context.cluster_id,
                )
                for t in raw_trades
            ]

        The generator should interrogate risk_factor.market_data to decide:
        - Strike grids (from vol surface ATM and wings)
        - Tenor grids (from rate curve liquid points)
        - Payoff types appropriate to the asset class
        """
        raise NotImplementedError(
            "Wire to your Generator.generate_for_factor() or equivalent"
        )

    @staticmethod
    def _deduplicate(trades: List[ElementaryTrade]) -> List[ElementaryTrade]:
        """Remove duplicate trades by trade_id, preserving first occurrence order."""
        seen: Set[str] = set()
        unique: List[ElementaryTrade] = []
        for t in trades:
            if t.trade_id not in seen:
                seen.add(t.trade_id)
                unique.append(t)
        return unique
```

---

## `replication/reduction/__init__.py`

**Path:** `src/rade_sr/replication/reduction/__init__.py`

```python
"""Dimensionality reduction for the elementary trade universe."""
```

---

## `replication/reduction/base.py`

**Path:** `src/rade_sr/replication/reduction/base.py`

```python
"""
Base protocol for dimensionality reduction strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class ReductionStrategy(ABC):
    """Abstract base for trade universe reduction strategies.

    All strategies take a PnL DataFrame (scenarios x trades) and return
    a reduced subset of trade IDs plus metadata about what was removed.
    """

    @abstractmethod
    def reduce(
        self,
        pnl_df: pd.DataFrame,
        n_target: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Reduce the trade universe.

        Parameters
        ----------
        pnl_df : pd.DataFrame
            PnL matrix, shape ``(n_scenarios, n_trades)``.
            Columns are trade IDs.
        n_target : int
            Target number of trades to keep.

        Returns
        -------
        dict
            - ``selected_trades``: list[str] — trade IDs to keep.
            - ``removed_trades``: list[str] — trade IDs removed.
            - ``explained_variance``: float (if applicable).
            - Any strategy-specific metadata.
        """
        ...
```

---

## `replication/reduction/grid.py`

**Path:** `src/rade_sr/replication/reduction/grid.py`

```python
"""
Grid-based dimensionality reduction.

Selects trades by uniformly sampling from a parameter grid
(e.g. every Nth strike, every Mth tenor). Simpler than KPCA but
preserves market-meaningful spacing.

TODO(wire): Adapt grid sampling to your trade parameter conventions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from src.rade_sr.replication.reduction.base import ReductionStrategy

logger = logging.getLogger(__name__)


class GridReduction(ReductionStrategy):
    """Reduce trade universe by uniform grid sampling.

    Parameters
    ----------
    stride : int
        Keep every ``stride``-th trade in sorted order.
    sort_by : str or None
        Trade parameter to sort by before striding.
    """

    def __init__(self, stride: int = 2, sort_by: str = "trade_id") -> None:
        self._stride = stride
        self._sort_by = sort_by

    def reduce(
        self,
        pnl_df: pd.DataFrame,
        n_target: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Select trades by grid sampling.

        TODO(wire): If your grid reduction needs parameter metadata
        (strike, tenor) to sort meaningfully, pass them via kwargs
        or modify the interface.
        """
        trade_ids = sorted(pnl_df.columns.tolist())
        selected = trade_ids[::self._stride][:n_target]
        removed = [tid for tid in trade_ids if tid not in set(selected)]

        logger.info("Grid reduction: %d -> %d trades", len(trade_ids), len(selected))

        return {
            "selected_trades": selected,
            "removed_trades": removed,
            "method": "grid",
        }
```

---

## `replication/reduction/kpca.py`

**Path:** `src/rade_sr/replication/reduction/kpca.py`

```python
"""
Kernel PCA-based dimensionality reduction.

Selects trades that best span the PnL space using Kernel PCA
decomposition. Trades are ranked by their contribution to the
principal components.

TODO(wire): This mirrors the KPCA reduction logic in your existing
dimension_reduction() function in rade_ml_pt. Wire or import as needed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA

from src.rade_sr.replication.reduction.base import ReductionStrategy

logger = logging.getLogger(__name__)


class KPCAReduction(ReductionStrategy):
    """Reduce trade universe via Kernel PCA.

    Parameters
    ----------
    kernel : str
        Kernel type: ``"rbf"``, ``"linear"``, ``"poly"``.
    gamma : float or None
        Kernel coefficient for rbf/poly.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        self._kernel = kernel
        self._gamma = gamma
        self._seed = seed

    def reduce(
        self,
        pnl_df: pd.DataFrame,
        n_target: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Select the top-n_target trades by KPCA importance.

        TODO(wire): If your existing KPCA reduction has additional logic
        (e.g. correlation filtering, minimum variance threshold), add it
        here or in a subclass.
        """
        trade_ids = pnl_df.columns.tolist()
        n_components = min(n_target, len(trade_ids), pnl_df.shape[0])

        kpca = KernelPCA(
            n_components=n_components,
            kernel=self._kernel,
            gamma=self._gamma,
            random_state=self._seed,
        )
        transformed = kpca.fit_transform(pnl_df.values.T)

        importance = np.linalg.norm(transformed, axis=1)
        ranked_indices = np.argsort(importance)[::-1]

        selected_indices = ranked_indices[:n_target]
        selected = [trade_ids[i] for i in sorted(selected_indices)]
        removed = [tid for tid in trade_ids if tid not in set(selected)]

        explained = float(np.sum(kpca.eigenvalues_[:n_components]) /
                          np.sum(kpca.eigenvalues_)) if hasattr(kpca, 'eigenvalues_') else 0.0

        logger.info(
            "KPCA reduction: %d -> %d trades (explained variance: %.2f%%)",
            len(trade_ids), len(selected), explained * 100,
        )

        return {
            "selected_trades": selected,
            "removed_trades": removed,
            "explained_variance": explained,
            "method": "kpca",
        }
```

---

## `replication/reduction/trade_reducer.py`

**Path:** `src/rade_sr/replication/reduction/trade_reducer.py`

```python
"""
Composite trade reducer — dispatches to the configured reduction strategy.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from src.rade_sr.replication.reduction.base import ReductionStrategy
from src.rade_sr.replication.reduction.kpca import KPCAReduction
from src.rade_sr.replication.reduction.grid import GridReduction

logger = logging.getLogger(__name__)

_STRATEGY_REGISTRY: Dict[str, type] = {
    "kpca": KPCAReduction,
    "grid": GridReduction,
}


class TradeReducer:
    """Reduces the elementary trade universe using a pluggable strategy.

    Parameters
    ----------
    method : str
        Reduction method: ``"kpca"`` or ``"grid"``.
    n_target : int
        Target number of trades to keep.
    strategy_kwargs : dict
        Additional keyword arguments passed to the strategy constructor.
    """

    def __init__(
        self,
        method: str = "kpca",
        n_target: int = 50,
        **strategy_kwargs: Any,
    ) -> None:
        self._n_target = n_target
        strategy_cls = _STRATEGY_REGISTRY.get(method)
        if strategy_cls is None:
            raise ValueError(
                f"Unknown reduction method={method!r}. "
                f"Available: {list(_STRATEGY_REGISTRY)}"
            )
        self._strategy: ReductionStrategy = strategy_cls(**strategy_kwargs)

    def reduce(self, pnl_df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """Run the reduction strategy.

        Parameters
        ----------
        pnl_df : pd.DataFrame
            PnL matrix (scenarios x trades).

        Returns
        -------
        dict
            ``selected_trades``, ``removed_trades``, and strategy metadata.
        """
        return self._strategy.reduce(pnl_df, self._n_target, **kwargs)
```

---

## `replication/trades/__init__.py`

**Path:** `src/rade_sr/replication/trades/__init__.py`

```python
"""Trade generation and PnL computation helpers."""
```

---

## `replication/trades/generator.py`

**Path:** `src/rade_sr/replication/trades/generator.py`

```python
"""
Core trade generation logic.

This is the concrete trade generation implementation that is wrapped
by RiskFactorAwareTradeGenerator. It contains the asset-class-specific
logic for creating elementary instruments from market data.

TODO(wire): This module holds your existing trade generation code.
The RiskFactorAwareTradeGenerator calls methods on this class to
produce raw instrument objects, which are then converted to
ElementaryTrade dataclasses.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class Generator:
    """Generates raw elementary instruments from risk factor market data.

    TODO(wire): This class wraps your existing trade generation logic.
    Replace the methods below with calls to your existing implementation.

    The RiskFactorAwareTradeGenerator calls:
        raw_trades = generator.generate_for_factor(
            factor_id=..., asset_class=..., market_data=...,
            scenarios=..., params=...,
        )

    Each raw trade should have:
        - .payoff_type (str)
        - .to_dict() -> dict of instrument parameters

    If your existing trades are already dicts, adapt accordingly.
    """

    def generate_for_factor(
        self,
        factor_id: str,
        asset_class: str,
        market_data: Any,
        scenarios: Any,
        params: Dict[str, Any],
    ) -> List[Any]:
        """Generate elementary instruments for one risk factor.

        Parameters
        ----------
        factor_id : str
            Risk factor identifier.
        asset_class : str
            Asset class (dispatches to appropriate generation logic).
        market_data : Any
            Loaded market data for this factor.
        scenarios : Any
            Shock scenarios for this factor.
        params : dict
            Trade generation parameters from PipelineConfig.trade_config.
            Typical keys: ``notional``, ``expiry_grid``, ``payoff_types``,
            ``n_strikes``, ``wing_width``.

        Returns
        -------
        list[Any]
            Raw trade objects. Each must have ``.payoff_type`` and ``.to_dict()``.

        TODO(wire): Replace with your existing generation dispatch:

            if asset_class == "fx":
                return self._generate_fx(factor_id, market_data, scenarios, params)
            elif asset_class == "rates":
                return self._generate_rates(factor_id, market_data, scenarios, params)
            ...

        Or better: use a registry pattern here too if generation logic
        is complex per asset class.
        """
        raise NotImplementedError(
            f"Wire trade generation for asset_class={asset_class!r}"
        )
```

---

## `replication/trades/pnl.py`

**Path:** `src/rade_sr/replication/trades/pnl.py`

```python
"""
Per-trade PnL helpers.

Utility functions for working with ElementaryTradePnL objects:
aggregation, filtering, matrix construction.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.rade_sr.core.types import ElementaryTrade, ElementaryTradePnL


def build_pnl_matrix(
    trades: List[ElementaryTrade],
    pnls: Dict[str, ElementaryTradePnL],
) -> np.ndarray:
    """Build a PnL matrix from trades and their PnL vectors.

    Parameters
    ----------
    trades : list[ElementaryTrade]
        Trades in the desired row order.
    pnls : dict[str, ElementaryTradePnL]
        PnL lookup by trade_id.

    Returns
    -------
    np.ndarray
        Shape ``(n_trades, n_scenarios)``.
    """
    return np.stack([pnls[t.trade_id].pnl_vector for t in trades])


def filter_pnls_by_cluster(
    pnls: Dict[str, ElementaryTradePnL],
    cluster_id: str,
) -> Dict[str, ElementaryTradePnL]:
    """Filter PnL dict to a single cluster."""
    return {
        tid: pnl for tid, pnl in pnls.items()
        if pnl.cluster_id == cluster_id
    }


def filter_pnls_by_factor(
    pnls: Dict[str, ElementaryTradePnL],
    factor_id: str,
) -> Dict[str, ElementaryTradePnL]:
    """Filter PnL dict to a single risk factor."""
    return {
        tid: pnl for tid, pnl in pnls.items()
        if pnl.factor_id == factor_id
    }
```

---

## `utils/__init__.py`

**Path:** `src/rade_sr/utils/__init__.py`

```python
"""Shared utility functions."""
```

---

## `utils/mapping.py`

**Path:** `src/rade_sr/utils/mapping.py`

```python
"""
ID and naming convention utilities.

Provides standardised naming for factors, trades, clusters, and
file paths used throughout the pipeline.

TODO(wire): Adapt conventions to match your existing naming standards.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def normalise_factor_id(raw_id: str) -> str:
    """Normalise a risk factor ID to a canonical form.

    Examples: ``"EUR/USD"`` → ``"EURUSD"``, ``"eur usd"`` → ``"EURUSD"``.

    TODO(wire): Adapt to your firm's factor ID conventions.
    """
    return raw_id.upper().replace("/", "").replace(" ", "").replace("_", "")


def normalise_ccy_pair(pair: str) -> str:
    """Normalise a currency pair to 6-character uppercase form.

    Examples: ``"eur/usd"`` → ``"EURUSD"``, ``"EUR-USD"`` → ``"EURUSD"``.
    """
    return pair.upper().replace("/", "").replace("-", "").replace(" ", "")


def build_cluster_id(
    method: str,
    key: Optional[str] = None,
    index: Optional[int] = None,
) -> str:
    """Build a standardised cluster ID string.

    Examples:
        ``build_cluster_id("kmeans", index=3)`` → ``"cluster_3"``
        ``build_cluster_id("user_defined", key="FX_FLOW")`` → ``"FX_FLOW"``
    """
    if method == "user_defined" and key:
        return key
    if index is not None:
        return f"cluster_{index}"
    raise ValueError("Either key or index must be provided")
```

---

## `utils/tools.py`

**Path:** `src/rade_sr/utils/tools.py`

```python
"""
General-purpose utility functions.

Timing, formatting, hashing, and other helpers used across layers.
"""
from __future__ import annotations

import hashlib
import json
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator

logger = logging.getLogger(__name__)


@contextmanager
def log_timer(label: str) -> Generator[None, None, None]:
    """Context manager that logs elapsed time for a block.

    Usage::

        with log_timer("Phase 1"):
            intermediates = pipeline._phase1(cluster_ids, config)
    """
    start = time.perf_counter()
    logger.info("[%s] started", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("[%s] completed in %.2fs", label, elapsed)


def dict_hash(d: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of a JSON-serialisable dict.

    Useful for cache keys and manifest bundle hashes.
    """
    serialised = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def chunked(items: list, chunk_size: int) -> list:
    """Split a list into chunks of at most chunk_size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
```

---
