# PORTING — Tier 1 copy-paste pack

Everything below is the **Tier 1** source of `rade_static_replication` — the files that
port to your environment **as-is** (pure domain, market data, pricing kernels, asset
plugins, pipeline, artifacts, config, and the mock/file-portfolio clients). Copy each
block into the matching path under your package root.

**Excluded on purpose (Tier 3 — you wire these per README §7):**
`clients/api/sage.py`, `clients/api/star.py`, `clients/file/marketdata.py`.
The mock market-data client below defines the exact array-shape contract those adapters
must return.

## Dependencies

```bash
pip install numpy pandas pyyaml pyarrow
pip install numba   # optional: JIT-compiles the pricing kernels; pure-Python fallback otherwise
```

## Run after porting

```bash
python -m <your_pkg_path>.examples.run_pipeline_mock
python -m pytest <your_pkg_path>/tests -q
```

## Files in this pack

- `__init__.py`
- `api.py`
- `pyproject.toml`
- `domain/__init__.py`
- `domain/contracts.py`
- `domain/enums.py`
- `domain/errors.py`
- `domain/instruments.py`
- `marketdata/__init__.py`
- `marketdata/base.py`
- `marketdata/common/__init__.py`
- `marketdata/common/curves.py`
- `marketdata/fx/__init__.py`
- `marketdata/fx/instruments.py`
- `marketdata/fx/scenarios.py`
- `marketdata/fx/snapshot.py`
- `marketdata/rates/__init__.py`
- `marketdata/rates/instruments.py`
- `marketdata/rates/scenarios.py`
- `marketdata/rates/snapshot.py`
- `marketdata/scenarios.py`
- `marketdata/shocks.py`
- `marketdata/snapshot.py`
- `pricing/__init__.py`
- `pricing/kernels/__init__.py`
- `pricing/kernels/_math.py`
- `pricing/kernels/_numba.py`
- `pricing/kernels/fx.py`
- `pricing/kernels/rates.py`
- `assets/__init__.py`
- `assets/base.py`
- `assets/fx/__init__.py`
- `assets/fx/builder.py`
- `assets/fx/generator.py`
- `assets/fx/instruments.py`
- `assets/fx/pricer.py`
- `assets/rates/__init__.py`
- `assets/rates/builder.py`
- `assets/rates/generator.py`
- `assets/rates/instruments.py`
- `assets/rates/pricer.py`
- `assets/registry.py`
- `portfolio/__init__.py`
- `portfolio/normalise.py`
- `portfolio/resolution/__init__.py`
- `portfolio/resolution/resolver.py`
- `portfolio/resolution/rules.py`
- `portfolio/validate.py`
- `config/__init__.py`
- `config/loader.py`
- `config/schema.py`
- `pipeline/__init__.py`
- `pipeline/context.py`
- `pipeline/engine/__init__.py`
- `pipeline/engine/pnl_engine.py`
- `pipeline/orchestrator.py`
- `pipeline/stages/__init__.py`
- `pipeline/stages/build_market.py`
- `pipeline/stages/cluster.py`
- `pipeline/stages/generate.py`
- `pipeline/stages/load.py`
- `pipeline/stages/normalise.py`
- `pipeline/stages/pnl.py`
- `pipeline/stages/price.py`
- `pipeline/stages/resolve.py`
- `artifacts/__init__.py`
- `artifacts/layout.py`
- `artifacts/manifest.py`
- `artifacts/store.py`
- `artifacts/writers.py`
- `clients/__init__.py`
- `clients/base.py`
- `clients/file/__init__.py`
- `clients/file/portfolio.py`
- `clients/mock/__init__.py`
- `clients/mock/marketdata.py`
- `clients/mock/portfolio.py`
- `clients/payloads.py`
- `configs/fx_mapping.csv`
- `configs/orchestrator.yaml`
- `examples/__init__.py`
- `examples/run_pipeline_mock.py`
- `tests/__init__.py`
- `tests/integration/__init__.py`
- `tests/integration/test_pipeline.py`
- `tests/unit/__init__.py`
- `tests/unit/test_marketdata.py`
- `tests/unit/test_pricing.py`

---

### `__init__.py`

```python
"""
rade_static_replication
========================

Preprocessing library that turns a raw derivatives portfolio (trade attributes +
historical scenario PnL) into the elementary-basis PnL tensors consumed by the
``rade_ml`` hybrid GNN-RNN model, using the theory of *static replication*.

Public entry points::

    from src.rade_static_replication import run, Orchestrator, OrchestratorConfig
    from src.rade_static_replication import load_config, default_registry

See :func:`run` / :mod:`api` for the one-call facade, or drive ``Orchestrator``
stage-by-stage for debugging. Architecture and module map are documented in
``README.md``.
"""
from __future__ import annotations

__version__ = "0.2.0"

from src.rade_static_replication.api import run
from src.rade_static_replication.assets.registry import default_registry
from src.rade_static_replication.config.loader import config_from_dict, load_config
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.pipeline.context import RunContext
from src.rade_static_replication.pipeline.orchestrator import Orchestrator

__all__ = [
    "__version__",
    "run",
    "Orchestrator",
    "RunContext",
    "OrchestratorConfig",
    "load_config",
    "config_from_dict",
    "default_registry",
]
```

### `api.py`

```python
"""
Public facade — one call to run the whole pipeline.

::

    from src.rade_static_replication import run, load_config
    from src.rade_static_replication.clients.mock import MockPortfolioClient, MockMarketDataClient

    ctx = run(load_config("configs/orchestrator.yaml"),
              MockPortfolioClient(), MockMarketDataClient())

Returns the populated :class:`RunContext`; artifacts are written to the configured
run directory. For step-by-step control, construct :class:`Orchestrator` directly.
"""
from __future__ import annotations

from typing import Optional

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.clients.base import MarketDataClient, PortfolioClient
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.pipeline.context import RunContext
from src.rade_static_replication.pipeline.orchestrator import Orchestrator


def run(
    config: OrchestratorConfig,
    portfolio_client: PortfolioClient,
    market_data_client: MarketDataClient,
    registry: Optional[Registry] = None,
) -> RunContext:
    """Run the full static-replication preprocessing pipeline."""
    return Orchestrator(config, portfolio_client, market_data_client, registry).run()
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "rade-static-replication"
version = "0.2.0"
description = "Static-replication preprocessing: portfolio -> elementary-basis PnL tensors for rade_ml."
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.23",
    "pandas>=1.5",
    "pyyaml>=6.0",
    "pyarrow>=10.0",
]

[project.optional-dependencies]
fast = ["numba>=0.57"]   # optional: JIT-compile the pricing kernels
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

### `domain/__init__.py`

```python
"""Pure domain layer: contracts, instruments, enums, errors. No I/O, no asset specifics."""
```

### `domain/contracts.py`

```python
"""
Stage contracts — the typed handoffs between pipeline stages.

The pipeline is a sequence of pure stages, each consuming the previous contract and
emitting exactly one new one:

    RawPortfolio -> Portfolio -> RiskFactorUniverse -> FactorDataSet
        -> ElementaryUniverse -> BasePriceSet -> PnLResult -> ClusterSet -> ArtifactManifest

Contracts are dumb dataclasses with light helpers — no I/O, no business logic. They
are the stable interface that lets any one stage be swapped without disturbing its
neighbours; the orchestrator records each on the ``RunContext``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.scenarios import ScenarioSet
from src.rade_static_replication.marketdata.snapshot import MarketSnapshot


# ----- portfolio -----

@dataclass(frozen=True)
class RawPortfolio:
    """Untouched portfolio payload from the portfolio client."""
    attributes: pd.DataFrame
    target_pnl: pd.DataFrame
    cob_date: str
    source: str = ""


@dataclass(frozen=True)
class Portfolio:
    """Normalised, validated portfolio — one row per trade."""
    attributes: pd.DataFrame
    target_pnl: pd.DataFrame
    scenario_ids: np.ndarray
    cob_date: str

    @property
    def trade_ids(self) -> List[str]:
        return [str(i) for i in self.attributes.index]

    @property
    def asset_classes(self) -> List[str]:
        if "AssetClass" not in self.attributes.columns:
            return []
        return sorted(self.attributes["AssetClass"].dropna().unique().tolist())


# ----- risk-factor universe -----

@dataclass(frozen=True)
class RiskFactorSpec:
    """Identity of a risk factor and its market-data dependencies."""
    factor_id: str
    asset_class: str
    dependencies: Tuple[str, ...] = ()
    is_primary: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskFactorUniverse:
    """De-duplicated set of risk factors the portfolio needs."""
    specs: Dict[str, RiskFactorSpec]
    factors_by_trade: Dict[str, List[str]]

    @property
    def primary_ids(self) -> List[str]:
        return [fid for fid, s in self.specs.items() if s.is_primary]

    def build_order(self) -> List[str]:
        """Dependency-respecting build order (topological sort)."""
        order: List[str] = []
        seen: set[str] = set()

        def _visit(fid: str, stack: Tuple[str, ...] = ()) -> None:
            if fid in seen:
                return
            if fid in stack:
                raise ValueError(f"cyclic risk-factor dependency at {fid}: {stack}")
            spec = self.specs.get(fid)
            if spec is not None:
                for dep in spec.dependencies:
                    _visit(dep, stack + (fid,))
            seen.add(fid)
            order.append(fid)

        for fid in self.specs:
            _visit(fid)
        return order


# ----- risk-factor data -----

@dataclass(frozen=True)
class RiskFactorData:
    """A factor with its built base snapshot + shocked scenario states."""
    spec: RiskFactorSpec
    snapshot: MarketSnapshot
    scenarios: ScenarioSet
    dependencies: Dict[str, "RiskFactorData"] = field(default_factory=dict)

    @property
    def factor_id(self) -> str:
        return self.spec.factor_id


@dataclass(frozen=True)
class FactorDataSet:
    """All built risk-factor data, keyed by ``factor_id``."""
    factors: Dict[str, RiskFactorData]

    @property
    def n_scenarios(self) -> int:
        for rf in self.factors.values():
            return rf.scenarios.n_scenarios
        return 0


# ----- elementary trades -----

@dataclass(frozen=True)
class ElementaryUniverse:
    """Elementary trades for the portfolio, grouped by ``factor_id``."""
    by_factor: Dict[str, List[ElementaryTrade]]

    def flat(self) -> List[ElementaryTrade]:
        out: List[ElementaryTrade] = []
        for fid in sorted(self.by_factor):
            out.extend(self.by_factor[fid])
        return out

    @property
    def n_trades(self) -> int:
        return sum(len(v) for v in self.by_factor.values())


# ----- pricing & PnL -----

@dataclass(frozen=True)
class BasePriceSet:
    """Base (COB) present values, ``factor_id -> {trade_id -> PV}``."""
    by_factor: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class FactorPnL:
    """Scenario PnL for one factor's elementary trades.

    ``pnl`` is ``(n_trades, n_scenarios)``, rows aligned to ``trade_ids`` and columns
    to ``scenario_ids``.
    """
    factor_id: str
    trade_ids: List[str]
    scenario_ids: np.ndarray
    pnl: np.ndarray


@dataclass(frozen=True)
class PnLResult:
    """All elementary PnL, ``factor_id -> FactorPnL``."""
    by_factor: Dict[str, FactorPnL]


# ----- clustering & artifacts -----

@dataclass(frozen=True)
class ClusterSpec:
    """A single-asset-class slice keyed by attribute values."""
    cluster_id: str
    key: Dict[str, str]
    asset_class: str
    risk_factor_ids: List[str]
    target_trade_ids: List[str]


@dataclass(frozen=True)
class ClusterSet:
    """The resolved list of clusters for the run."""
    clusters: List[ClusterSpec]


@dataclass(frozen=True)
class ArtifactPaths:
    """The four per-cluster files consumed by ``rade_ml``."""
    elementary_pnl: Path
    elementary_attributes: Path
    target_pnl: Path
    target_attributes: Path


@dataclass(frozen=True)
class ArtifactManifest:
    """Run-level manifest: every cluster's paths + metadata (the ``jobs.pkl`` payload)."""
    entries: List[Dict[str, Any]]
    root: Path
```

### `domain/enums.py`

```python
"""
Controlled vocabularies used across the library.

String-valued enums so they serialise cleanly into configs, manifests, and
artifact metadata while still giving type-safety and a single definition site.
"""
from __future__ import annotations

from enum import Enum


class AssetClass(str, Enum):
    """Built-in asset classes (extend by registering a new plugin)."""
    FX = "fx"
    RATES = "rates"


class PayoffType(str, Enum):
    """Elementary-trade payoff types."""
    CALL = "call"
    PUT = "put"
    DIGITAL_CALL = "digital_call"
    DIGITAL_PUT = "digital_put"
    FORWARD = "forward"
    PAYER = "payer"
    RECEIVER = "receiver"
    SWAP = "swap"


class StrikeConvention(str, Enum):
    """How a vol surface's strike axis is expressed."""
    ABSOLUTE = "absolute"
    MONEYNESS = "moneyness"   # K / forward
    DELTA = "delta"


class VolType(str, Enum):
    LOGNORMAL = "lognormal"   # Black / Garman-Kohlhagen
    NORMAL = "normal"         # Bachelier (bp vol)


class DayCount(str, Enum):
    ACT_365 = "act/365"
    ACT_360 = "act/360"


class Interpolation(str, Enum):
    """Curve interpolation rules."""
    LINEAR_ZERO = "linear_zero"          # linear on the zero rate
    LOG_LINEAR_DF = "log_linear_df"      # log-linear on discount factors (flat fwd)


class ShockMode(str, Enum):
    """How a scenario shock maps onto the COB base level."""
    ABSOLUTE = "absolute"          # scenario value IS the level
    ADDITIVE = "additive"          # level = base + shock
    RELATIVE = "relative"          # level = base * (1 + shock)
    LOG_RELATIVE = "log_relative"  # level = base * exp(shock)
```

### `domain/errors.py`

```python
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
```

### `domain/instruments.py`

```python
"""
Elementary instrument definition.

An :class:`ElementaryTrade` is one member of the replicating basis. It is
asset-class-agnostic on purpose: the asset plugin's generator decides *which*
trades to emit and its pricer knows how to read ``parameters``. The ``trade_id`` is
``|``-delimited so the downstream consumer can recover the underlying via
``id.split("|")``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ElementaryTrade:
    """One elementary (replicating-basis) instrument.

    Parameters
    ----------
    trade_id : str
        Unique id, e.g. ``"FX.SPOT.USD.GBP|CALL|K=1.25|T=0.50"``.
    factor_id : str
        Risk factor this instrument is priced against.
    asset_class : str
        Pricer routing key.
    payoff_type : str
        Instrument type (see :class:`~...domain.enums.PayoffType`).
    parameters : dict
        Pricer inputs (must include ``expiry`` in years).
    notional : float
        Unit notional by default — the model learns replicating weights.
    """
    trade_id: str
    factor_id: str
    asset_class: str
    payoff_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    notional: float = 1.0
```

### `marketdata/__init__.py`

```python
"""
Typed market-data layer.

Immutable, self-validating objects with explicit conventions — the single source of
truth for "what the market looked like" per risk factor. No nested dicts.

Layout mirrors ``assets/``:

* ``base`` / ``snapshot`` / ``scenarios`` — the shared spine + abstract bases;
* ``common/`` — the one cross-asset instrument (``DiscountCurve``);
* ``fx/`` and ``rates/`` — each asset class's own instruments + snapshot + scenario set
  (FX 2-D lognormal surface vs rates 3-D normal cube — different shapes/conventions);
* ``shocks`` — relative/absolute shock resolution.
"""
from src.rade_static_replication.marketdata.common import DiscountCurve
from src.rade_static_replication.marketdata.fx import FXScenarioSet, FXSnapshot, Spot, VolSurface
from src.rade_static_replication.marketdata.rates import (
    RatesScenarioSet,
    RatesSnapshot,
    VolCube,
)
from src.rade_static_replication.marketdata.scenarios import ScenarioSet
from src.rade_static_replication.marketdata.shocks import ShockConvention, apply_shock
from src.rade_static_replication.marketdata.snapshot import MarketSnapshot

__all__ = [
    "DiscountCurve", "VolSurface", "VolCube", "Spot",
    "MarketSnapshot", "FXSnapshot", "RatesSnapshot",
    "ScenarioSet", "FXScenarioSet", "RatesScenarioSet",
    "ShockConvention", "apply_shock",
]
```

### `marketdata/base.py`

```python
"""
Foundations for the typed market-data layer.

This is the "stronger base" the market objects build on:

* :class:`Interpolator` — a pluggable interpolation strategy (linear today, with a
  clear seam for log-linear-DF / variance / SABR) so a desk can change interpolation
  policy in one place without touching pricers.
* validation helpers (:func:`require_increasing`, :func:`require_same_shape`, …) that
  give every object fail-fast, self-describing ``__post_init__`` checks.
* :class:`MarketObject` — the structural protocol every market object satisfies
  (``label`` + ``validate`` + ``summary``), used for typing and audit logging.
* :func:`year_fraction` — the single day-count implementation.

Keeping all of this here means the curves/surfaces/cubes stay small and consistent.
"""
from __future__ import annotations

import datetime as _dt
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from src.rade_static_replication.domain.enums import DayCount
from src.rade_static_replication.domain.errors import MarketDataError


# =====================================================================
#  Interpolation strategies
# =====================================================================

@runtime_checkable
class Interpolator(Protocol):
    """Maps a query point onto sampled ``(xp, fp)`` data."""

    def __call__(self, x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        ...


class LinearInterpolator:
    """Piecewise-linear with flat extrapolation (the safe default)."""

    def __call__(self, x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        return float(np.interp(x, xp, fp))


DEFAULT_INTERPOLATOR: Interpolator = LinearInterpolator()


def total_variance_interp(
    x: float, xp: np.ndarray, vols_at_x: np.ndarray, axis_vals: np.ndarray,
) -> float:
    """Interpolate a vol at expiry ``x`` in *total variance* (the arb-aware default).

    ``vols_at_x`` are the per-expiry vols (already strike-interpolated) on the
    ``axis_vals`` expiry pillars.
    """
    if x <= axis_vals[0]:
        return float(vols_at_x[0])
    if x >= axis_vals[-1]:
        return float(vols_at_x[-1])
    j = int(np.searchsorted(axis_vals, x))
    t0, t1 = axis_vals[j - 1], axis_vals[j]
    w = (x - t0) / (t1 - t0)
    var = (1.0 - w) * vols_at_x[j - 1] ** 2 * t0 + w * vols_at_x[j] ** 2 * t1
    return float(np.sqrt(max(var, 0.0) / x))


# =====================================================================
#  Validation helpers
# =====================================================================

def as_1d(name: str, arr: Sequence) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise MarketDataError(f"{name}: expected non-empty 1-D array, got shape {a.shape}")
    return a


def require_increasing(name: str, arr: np.ndarray) -> None:
    if arr.size > 1 and np.any(np.diff(arr) <= 0):
        raise MarketDataError(f"{name}: values must be strictly increasing")


def require_same_shape(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise MarketDataError(f"{name_a} {a.shape} != {name_b} {b.shape}")


def require_shape(name: str, a: np.ndarray, expected: tuple) -> None:
    if a.shape != expected:
        raise MarketDataError(f"{name} shape {a.shape} != {expected}")


def require_positive(name: str, value: float) -> None:
    if not (value > 0.0):
        raise MarketDataError(f"{name} must be positive, got {value}")


# =====================================================================
#  Day count
# =====================================================================

def year_fraction(start: _dt.date, end: _dt.date, convention: DayCount = DayCount.ACT_365) -> float:
    """Year fraction between two dates (non-negative)."""
    days = max((end - start).days, 0)
    basis = 360.0 if convention == DayCount.ACT_360 else 365.0
    return days / basis


# =====================================================================
#  Structural protocol
# =====================================================================

@runtime_checkable
class MarketObject(Protocol):
    """Everything a market object exposes for typing + audit."""
    label: str

    def validate(self) -> None: ...

    def summary(self) -> dict: ...
```

### `marketdata/common/__init__.py`

```python
"""Cross-asset market instruments (the discount curve, shared by every asset class)."""
from src.rade_static_replication.marketdata.common.curves import DiscountCurve

__all__ = ["DiscountCurve"]
```

### `marketdata/common/curves.py`

```python
"""
Discount / zero-rate curve — the one genuinely cross-asset market instrument.

A ``DiscountCurve`` is the same object whether an FX factor (which needs two of them) or a
rates factor consumes it, so it lives in ``common`` rather than any asset subpackage.
Stores continuously-compounded zero rates on year-fraction pillars and exposes the
quantities pricers need — ``df``, ``zero``, ``forward_rate`` — under a chosen
interpolation policy. Immutable; mutate via the ``with_*`` helpers (used by the shock
layer to produce scenario curves).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.rade_static_replication.domain.enums import Interpolation
from src.rade_static_replication.domain.errors import MarketDataError
from src.rade_static_replication.marketdata.base import (
    as_1d,
    require_increasing,
    require_same_shape,
)


@dataclass(frozen=True)
class DiscountCurve:
    """A continuously-compounded zero curve for one currency.

    Parameters
    ----------
    currency : str
    tenors : np.ndarray
        Pillar tenors in year fractions, strictly increasing.
    zero_rates : np.ndarray
        Continuously-compounded zero rates at each pillar.
    interpolation : Interpolation
        ``LINEAR_ZERO`` (default) or ``LOG_LINEAR_DF`` (flat-forward).
    """
    currency: str
    tenors: np.ndarray
    zero_rates: np.ndarray
    interpolation: Interpolation = Interpolation.LINEAR_ZERO
    label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = as_1d(f"{self.currency} curve tenors", self.tenors)
        z = as_1d(f"{self.currency} curve zero_rates", self.zero_rates)
        require_same_shape("tenors", t, "zero_rates", z)
        require_increasing(f"{self.currency} curve tenors", t)
        object.__setattr__(self, "tenors", t)
        object.__setattr__(self, "zero_rates", z)
        object.__setattr__(self, "label", self.label or f"DF.{self.currency}")

    # ---- core quantities ----

    def df(self, t: float) -> float:
        """Discount factor to ``t`` (year fraction)."""
        if t <= 0.0:
            return 1.0
        if self.interpolation == Interpolation.LOG_LINEAR_DF:
            log_df_pillars = -self.zero_rates * self.tenors
            return float(np.exp(np.interp(t, self.tenors, log_df_pillars)))
        return float(np.exp(-self.zero(t) * t))

    def zero(self, t: float) -> float:
        """Continuously-compounded zero rate at ``t``."""
        if t <= 0.0:
            return float(self.zero_rates[0])
        if self.interpolation == Interpolation.LOG_LINEAR_DF:
            return float(-np.log(self.df(t)) / t)
        return float(np.interp(t, self.tenors, self.zero_rates))

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously-compounded forward rate between ``t1`` and ``t2``."""
        if t2 <= t1:
            raise MarketDataError(f"forward_rate needs t2 > t1, got t1={t1}, t2={t2}")
        return float((np.log(self.df(t1)) - np.log(self.df(t2))) / (t2 - t1))

    # ---- mutation helpers (return new curves) ----

    def with_zero_rates(self, zero_rates: np.ndarray) -> "DiscountCurve":
        """A copy with replaced zero rates on the same pillars."""
        return DiscountCurve(
            self.currency, self.tenors.copy(), np.asarray(zero_rates, dtype=np.float64),
            self.interpolation, self.label, dict(self.metadata),
        )

    def validate(self) -> None:  # already validated in __post_init__
        return None

    def summary(self) -> dict:
        return {
            "type": "DiscountCurve", "currency": self.currency,
            "n_pillars": int(self.tenors.size),
            "tenor_range": [float(self.tenors[0]), float(self.tenors[-1])],
            "interpolation": self.interpolation.value,
        }
```

### `marketdata/fx/__init__.py`

```python
"""FX market-data objects: instruments (Spot, VolSurface) + snapshot + scenario set."""
from src.rade_static_replication.marketdata.fx.instruments import Spot, VolSurface
from src.rade_static_replication.marketdata.fx.scenarios import FXScenarioSet
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot

__all__ = ["Spot", "VolSurface", "FXSnapshot", "FXScenarioSet"]
```

### `marketdata/fx/instruments.py`

```python
"""
FX market instruments — spot quote and the FX implied-vol surface.

These are FX-specific in both shape and convention: the surface is 2-D in
(expiry, strike-axis) with an FX strike convention (moneyness / delta / absolute) and
lognormal vols. Rates' instrument (the 3-D normal-vol cube) lives in
``marketdata/rates/instruments.py`` — different dimensions, different convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.rade_static_replication.domain.enums import StrikeConvention, VolType
from src.rade_static_replication.marketdata.base import (
    as_1d,
    require_increasing,
    require_positive,
    require_shape,
    total_variance_interp,
)


@dataclass(frozen=True)
class Spot:
    """FX spot, quoted domestic-per-foreign for an ``XXXYYY`` pair."""
    pair: str
    value: float

    def __post_init__(self) -> None:
        require_positive(f"{self.pair} spot", self.value)

    @property
    def label(self) -> str:
        return f"FX.SPOT.{self.pair}"

    def validate(self) -> None:
        return None

    def summary(self) -> dict:
        return {"type": "Spot", "pair": self.pair, "value": float(self.value)}


@dataclass(frozen=True)
class VolSurface:
    """FX implied-vol surface ``vols[expiry, strike]``.

    Interpolates linearly in the strike axis and in **total variance** along expiry (the
    no-arbitrage-leaning default). A SABR/SVI upgrade slots in behind :meth:`vol`.

    Parameters
    ----------
    expiries : np.ndarray
        Expiry tenors (years), strictly increasing.
    strikes : np.ndarray
        Strike axis (units per ``strike_convention``), strictly increasing.
    vols : np.ndarray
        Implied vols, shape ``(n_exp, n_k)``.
    strike_convention : StrikeConvention
    vol_type : VolType
    """
    expiries: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    strike_convention: StrikeConvention = StrikeConvention.MONEYNESS
    vol_type: VolType = VolType.LOGNORMAL
    label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        exp = as_1d("vol expiries", self.expiries)
        k = as_1d("vol strikes", self.strikes)
        v = np.asarray(self.vols, dtype=np.float64)
        require_shape("vols", v, (exp.size, k.size))
        require_increasing("vol expiries", exp)
        object.__setattr__(self, "expiries", exp)
        object.__setattr__(self, "strikes", k)
        object.__setattr__(self, "vols", v)
        object.__setattr__(self, "strike_convention", StrikeConvention(self.strike_convention))
        object.__setattr__(self, "vol_type", VolType(self.vol_type))

    def vol(self, strike: float, expiry: float) -> float:
        """Interpolated vol at ``(strike, expiry)`` (strike units per convention)."""
        vols_by_exp = np.array([
            np.interp(strike, self.strikes, self.vols[i]) for i in range(self.expiries.size)
        ])
        return total_variance_interp(expiry, self.strikes, vols_by_exp, self.expiries)

    def validate(self) -> None:
        return None

    def summary(self) -> dict:
        return {
            "type": "VolSurface",
            "n_expiries": int(self.expiries.size), "n_strikes": int(self.strikes.size),
            "strike_convention": self.strike_convention.value, "vol_type": self.vol_type.value,
        }
```

### `marketdata/fx/scenarios.py`

```python
"""FX scenario set — shocked FX states aligned to an FXSnapshot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.rade_static_replication.domain.errors import MarketDataError
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot
from src.rade_static_replication.marketdata.scenarios import ScenarioSet


@dataclass(frozen=True)
class FXScenarioSet(ScenarioSet):
    """Shocked FX states (absolute levels) aligned to an :class:`FXSnapshot`."""
    spot: np.ndarray = None  # type: ignore[assignment]  # (n,)
    vol: np.ndarray = None                                # (n, n_exp, n_k)
    vol_expiries: np.ndarray = None
    vol_strikes: np.ndarray = None
    domestic_rate: np.ndarray = None                      # (n, n_dom)
    domestic_tenors: np.ndarray = None
    foreign_rate: np.ndarray = None                       # (n, n_for)
    foreign_tenors: np.ndarray = None

    def validate_against(self, snap: FXSnapshot) -> None:
        n = self.n_scenarios
        problems: List[str] = []
        if self.spot.shape != (n,):
            problems.append(f"spot {self.spot.shape} != ({n},)")
        if self.vol.shape != (n, self.vol_expiries.size, self.vol_strikes.size):
            problems.append(
                f"vol {self.vol.shape} != ({n}, {self.vol_expiries.size}, {self.vol_strikes.size})"
            )
        if self.domestic_rate.shape != (n, self.domestic_tenors.size):
            problems.append(f"domestic_rate {self.domestic_rate.shape} != ({n}, {self.domestic_tenors.size})")
        if self.foreign_rate.shape != (n, self.foreign_tenors.size):
            problems.append(f"foreign_rate {self.foreign_rate.shape} != ({n}, {self.foreign_tenors.size})")
        if problems:
            raise MarketDataError(f"{self.factor_id} FX scenario set misaligned: " + "; ".join(problems))
```

### `marketdata/fx/snapshot.py`

```python
"""FX market snapshot — base market for one USD-facing pair."""
from __future__ import annotations

from dataclasses import dataclass

from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.fx.instruments import Spot, VolSurface
from src.rade_static_replication.marketdata.snapshot import MarketSnapshot


@dataclass(frozen=True)
class FXSnapshot(MarketSnapshot):
    """Base FX market for one USD-facing pair (e.g. ``EURUSD``)."""
    spot: Spot = None  # type: ignore[assignment]
    domestic_curve: DiscountCurve = None  # numeraire / quote ccy (e.g. USD)
    foreign_curve: DiscountCurve = None   # asset / base ccy (e.g. EUR)
    vol_surface: VolSurface = None

    def forward(self, expiry: float) -> float:
        """Outright FX forward via covered interest parity: ``S * DF_f / DF_d``."""
        return self.spot.value * self.foreign_curve.df(expiry) / self.domestic_curve.df(expiry)
```

### `marketdata/rates/__init__.py`

```python
"""Rates market-data objects: instrument (VolCube) + snapshot + scenario set."""
from src.rade_static_replication.marketdata.rates.instruments import VolCube
from src.rade_static_replication.marketdata.rates.scenarios import RatesScenarioSet
from src.rade_static_replication.marketdata.rates.snapshot import RatesSnapshot

__all__ = ["VolCube", "RatesSnapshot", "RatesScenarioSet"]
```

### `marketdata/rates/instruments.py`

```python
"""
Rates market instrument — the swaption volatility cube.

Rates-specific in shape and convention: a 3-D cube in (option-expiry, swap-tenor, strike)
holding **normal** (bp) vols for Bachelier pricing — distinct from the 2-D lognormal FX
surface in ``marketdata/fx/instruments.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.rade_static_replication.domain.enums import VolType
from src.rade_static_replication.marketdata.base import as_1d, require_shape


@dataclass(frozen=True)
class VolCube:
    """Swaption vols ``vols[expiry, swap_tenor, strike]`` (normal/bp by default)."""
    expiries: np.ndarray
    swap_tenors: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    vol_type: VolType = VolType.NORMAL
    label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        e = as_1d("cube expiries", self.expiries)
        s = as_1d("cube swap_tenors", self.swap_tenors)
        k = as_1d("cube strikes", self.strikes)
        v = np.asarray(self.vols, dtype=np.float64)
        require_shape("cube vols", v, (e.size, s.size, k.size))
        object.__setattr__(self, "expiries", e)
        object.__setattr__(self, "swap_tenors", s)
        object.__setattr__(self, "strikes", k)
        object.__setattr__(self, "vols", v)
        object.__setattr__(self, "vol_type", VolType(self.vol_type))

    def vol(self, expiry: float, swap_tenor: float, strike: float) -> float:
        """Trilinear interpolation at ``(expiry, swap_tenor, strike)``."""
        ie = np.interp(expiry, self.expiries, np.arange(self.expiries.size))
        it = np.interp(swap_tenor, self.swap_tenors, np.arange(self.swap_tenors.size))
        ik = np.interp(strike, self.strikes, np.arange(self.strikes.size))

        def _lerp(arr: np.ndarray, idx: float) -> np.ndarray:
            lo = int(np.floor(idx))
            hi = min(lo + 1, arr.shape[0] - 1)
            w = idx - lo
            return (1.0 - w) * arr[lo] + w * arr[hi]

        return float(_lerp(_lerp(_lerp(self.vols, ie), it), ik))

    def validate(self) -> None:
        return None

    def summary(self) -> dict:
        return {
            "type": "VolCube",
            "shape": [int(self.expiries.size), int(self.swap_tenors.size), int(self.strikes.size)],
            "vol_type": self.vol_type.value,
        }
```

### `marketdata/rates/scenarios.py`

```python
"""Rates scenario set — shocked curve (+ cube) states aligned to a RatesSnapshot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.rade_static_replication.domain.errors import MarketDataError
from src.rade_static_replication.marketdata.rates.snapshot import RatesSnapshot
from src.rade_static_replication.marketdata.scenarios import ScenarioSet


@dataclass(frozen=True)
class RatesScenarioSet(ScenarioSet):
    """Shocked rates states (absolute levels) aligned to a :class:`RatesSnapshot`."""
    curve: np.ndarray = None  # type: ignore[assignment]  # (n, n_pillars)
    curve_tenors: np.ndarray = None
    vol: Optional[np.ndarray] = None        # (n, n_exp, n_ten, n_k)
    vol_expiries: Optional[np.ndarray] = None
    vol_swap_tenors: Optional[np.ndarray] = None
    vol_strikes: Optional[np.ndarray] = None

    def validate_against(self, snap: RatesSnapshot) -> None:
        n = self.n_scenarios
        problems: List[str] = []
        if self.curve.shape != (n, self.curve_tenors.size):
            problems.append(f"curve {self.curve.shape} != ({n}, {self.curve_tenors.size})")
        if self.vol is not None:
            expected = (n, self.vol_expiries.size, self.vol_swap_tenors.size, self.vol_strikes.size)
            if self.vol.shape != expected:
                problems.append(f"vol {self.vol.shape} != {expected}")
        if problems:
            raise MarketDataError(f"{self.factor_id} rates scenario set misaligned: " + "; ".join(problems))
```

### `marketdata/rates/snapshot.py`

```python
"""Rates market snapshot — base market for one currency curve (+ optional cube)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.rates.instruments import VolCube
from src.rade_static_replication.marketdata.snapshot import MarketSnapshot


@dataclass(frozen=True)
class RatesSnapshot(MarketSnapshot):
    """Base rates market for one currency curve (+ optional swaption cube)."""
    discount_curve: DiscountCurve = None  # type: ignore[assignment]
    vol_cube: Optional[VolCube] = None
```

### `marketdata/scenarios.py`

```python
"""
Scenario-set base.

A scenario set holds the shocked market states (resolved to absolute levels) for one
risk factor, aligned to an ordered ``scenario_ids`` sequence. The abstract base lives
here; asset-class-specific sets (``FXScenarioSet``, ``RatesScenarioSet``) live in
``marketdata/fx`` and ``marketdata/rates``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioSet:
    """Base class for shocked market states for one risk factor."""
    factor_id: str
    asset_class: str
    scenario_ids: np.ndarray  # (n_scenarios,)

    @property
    def n_scenarios(self) -> int:
        return int(np.asarray(self.scenario_ids).shape[0])
```

### `marketdata/shocks.py`

```python
"""
Shock application — resolve raw shocks into absolute market states.

Your environment expresses some shocks **relatively** and others **absolutely**.
That convention is a property of *how the data was produced*, so it is declared
here (per field) and applied at the client/builder boundary. Everything downstream
then sees only resolved absolute levels.

Usage::

    conv = ShockConvention({"spot": ShockMode.LOG_RELATIVE, "rate": ShockMode.ADDITIVE})
    spot_states = conv.apply("spot", base_spot, raw_spot_shock)   # (n_scenarios, ...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from src.rade_static_replication.domain.enums import ShockMode


def apply_shock(base: np.ndarray | float, shock: np.ndarray, mode: ShockMode) -> np.ndarray:
    """Resolve a raw ``shock`` against ``base`` into absolute levels.

    ``base`` broadcasts against ``shock`` (e.g. base curve ``(n_pillars,)`` vs shock
    ``(n_scenarios, n_pillars)``).
    """
    base = np.asarray(base, dtype=np.float64)
    shock = np.asarray(shock, dtype=np.float64)
    if mode == ShockMode.ABSOLUTE:
        return shock
    if mode == ShockMode.ADDITIVE:
        return base + shock
    if mode == ShockMode.RELATIVE:
        return base * (1.0 + shock)
    if mode == ShockMode.LOG_RELATIVE:
        return base * np.exp(shock)
    raise ValueError(f"unknown shock mode {mode!r}")


@dataclass(frozen=True)
class ShockConvention:
    """Per-field shock modes with a default fallback."""
    modes: Mapping[str, ShockMode] = field(default_factory=dict)
    default: ShockMode = ShockMode.ABSOLUTE

    def mode_for(self, field_name: str) -> ShockMode:
        return self.modes.get(field_name, self.default)

    def apply(self, field_name: str, base: np.ndarray | float, shock: np.ndarray) -> np.ndarray:
        return apply_shock(base, shock, self.mode_for(field_name))
```

### `marketdata/snapshot.py`

```python
"""
Market snapshot base.

A snapshot is the consistent COB market for one risk factor. The abstract base lives
here; asset-class-specific snapshots (``FXSnapshot``, ``RatesSnapshot``) live in
``marketdata/fx`` and ``marketdata/rates`` — mirroring the ``assets/`` layout.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSnapshot:
    """Base class for an asset-class-specific COB market snapshot."""
    factor_id: str
    asset_class: str
    as_of: _dt.date

    def summary(self) -> dict:
        return {"factor_id": self.factor_id, "asset_class": self.asset_class, "as_of": str(self.as_of)}
```

### `pricing/__init__.py`

```python
"""Asset-agnostic pricing numerics (the compiled kernels)."""
```

### `pricing/kernels/__init__.py`

```python
"""Compiled pricing kernels (FX + rates). Primitive in, primitive out."""
```

### `pricing/kernels/_math.py`

```python
"""
Shared numeric utilities for pricing kernels — all ``@njit`` compiled.

Normal CDF/PDF via ``erfc`` so the entire pricing path stays in compiled code with
no scipy dependency inside the JIT boundary.
"""
from __future__ import annotations

import math

import numpy as np
from src.rade_static_replication.pricing.kernels._numba import njit, float64


@njit(float64(float64), cache=True)
def norm_cdf(x: float) -> float:
    """Standard normal CDF via the complementary error function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@njit(float64(float64), cache=True)
def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@njit(float64(float64, float64), cache=True)
def df_from_rate(rate: float, t: float) -> float:
    """Continuous discount factor ``exp(-r t)``."""
    return math.exp(-rate * t)


@njit(cache=True)
def compute_annuity(tenors: np.ndarray, values: np.ndarray, t_start: float, tenor: float, freq: float) -> float:
    """Swap annuity from curve pillars (linear interp on the zero rate per pay date)."""
    n_periods = int(tenor / freq)
    annuity = 0.0
    for i in range(1, n_periods + 1):
        t_i = t_start + i * freq
        r_i = np.interp(t_i, tenors, values)
        annuity += math.exp(-r_i * t_i) * freq
    return annuity


@njit(cache=True)
def compute_forward_swap_rate(tenors: np.ndarray, values: np.ndarray, t_start: float, tenor: float, freq: float) -> float:
    """Forward par swap rate ``(DF(t_start) - DF(t_end)) / annuity``."""
    r_start = np.interp(t_start, tenors, values)
    r_end = np.interp(t_start + tenor, tenors, values)
    df_start = math.exp(-r_start * t_start)
    df_end = math.exp(-r_end * (t_start + tenor))
    annuity = compute_annuity(tenors, values, t_start, tenor, freq)
    if annuity == 0.0:
        return 0.0
    return (df_start - df_end) / annuity
```

### `pricing/kernels/_numba.py`

```python
"""
Optional-Numba shim.

Where Numba is installed this re-exports the real symbols; where it is not, it
provides no-op fallbacks so kernels still run as plain Python/NumPy. Always import
``njit``/type symbols from here, never from ``numba`` directly.
"""
from __future__ import annotations

try:  # pragma: no cover - depends on environment
    from numba import boolean, float64, int64, njit, prange, vectorize  # type: ignore

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    def vectorize(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

    prange = range  # type: ignore

    class _NumbaType:
        def __call__(self, *args, **kwargs):
            return self

        def __getitem__(self, item):
            return self

    float64 = _NumbaType()  # type: ignore
    int64 = _NumbaType()  # type: ignore
    boolean = _NumbaType()  # type: ignore
```

### `pricing/kernels/fx.py`

```python
"""
FX pricing kernels — Garman-Kohlhagen, ``@njit`` compiled.

Two flavours per instrument:

* **scalar** kernels (``price_fx_*``) price one trade in one market state;
* **vectorised** kernels (``price_fx_*_vec``) price one trade across a whole scenario
  window, looping in compiled code.

Conventions: ``spot`` is domestic-per-foreign, ``rate_dom`` the domestic (quote-ccy)
continuously-compounded rate, ``rate_for`` the foreign (base-ccy) rate. The foreign
currency behaves like a continuous dividend yield, hence the dual discounting. Inputs
are primitives / NumPy arrays only so the whole call stays inside the JIT boundary.
"""
from __future__ import annotations

import math

import numpy as np
from src.rade_static_replication.pricing.kernels._numba import njit
from src.rade_static_replication.pricing.kernels._math import norm_cdf, df_from_rate


@njit(cache=True)
def price_fx_forward(spot, strike, expiry, rate_dom, rate_for, notional, direction):
    """PV of an FX forward.

    ``direction`` is +1 for a long (buy-foreign) forward, -1 for short. The value is the
    discounted difference between the delivered spot and the agreed strike.
    """
    foreign_leg = spot * df_from_rate(rate_for, expiry)   # PV of receiving 1 unit foreign
    strike_leg = strike * df_from_rate(rate_dom, expiry)  # PV of paying the strike
    return direction * notional * (foreign_leg - strike_leg)


@njit(cache=True)
def price_fx_vanilla(spot, strike, expiry, rate_dom, rate_for, vol, is_call, notional):
    """Garman-Kohlhagen European FX option PV."""
    # Degenerate market: no time value left, fall back to (undiscounted) intrinsic.
    if expiry <= 0.0 or vol <= 0.0:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return notional * intrinsic

    sqrt_t = math.sqrt(expiry)
    # Standard GK d1/d2 with the (rate_dom - rate_for) drift.
    d1 = (math.log(spot / strike) + (rate_dom - rate_for + 0.5 * vol * vol) * expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df_dom = df_from_rate(rate_dom, expiry)
    df_for = df_from_rate(rate_for, expiry)

    if is_call:
        return notional * (spot * df_for * norm_cdf(d1) - strike * df_dom * norm_cdf(d2))
    return notional * (strike * df_dom * norm_cdf(-d2) - spot * df_for * norm_cdf(-d1))


@njit(cache=True)
def price_fx_digital(spot, strike, expiry, rate_dom, rate_for, vol, is_call, payout):
    """Cash-or-nothing digital: pays ``payout`` (domestic) if S_T is beyond the strike."""
    df_dom = df_from_rate(rate_dom, expiry)
    # Degenerate market: pay the (discounted) payout iff currently in-the-money.
    if expiry <= 0.0 or vol <= 0.0:
        if is_call:
            return payout * df_dom if spot > strike else 0.0
        return payout * df_dom if spot < strike else 0.0

    sqrt_t = math.sqrt(expiry)
    # Probability of finishing ITM under Q is N(d2) (call) / N(-d2) (put).
    d2 = (math.log(spot / strike) + (rate_dom - rate_for - 0.5 * vol * vol) * expiry) / (vol * sqrt_t)
    return payout * df_dom * (norm_cdf(d2) if is_call else norm_cdf(-d2))


@njit(cache=True)
def price_fx_vanilla_vec(spot_path, strike, expiry, rate_dom_path, rate_for_path, vol_path, is_call, notional):
    """PV of one vanilla across ``n`` scenarios; market inputs are ``(n,)`` arrays."""
    n = spot_path.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_fx_vanilla(
            spot_path[i], strike, expiry, rate_dom_path[i], rate_for_path[i], vol_path[i], is_call, notional,
        )
    return pv


@njit(cache=True)
def price_fx_digital_vec(spot_path, strike, expiry, rate_dom_path, rate_for_path, vol_path, is_call, payout):
    """PV of one digital across ``n`` scenarios."""
    n = spot_path.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_fx_digital(
            spot_path[i], strike, expiry, rate_dom_path[i], rate_for_path[i], vol_path[i], is_call, payout,
        )
    return pv


@njit(cache=True)
def price_fx_forward_vec(spot_path, strike, expiry, rate_dom_path, rate_for_path, notional, direction):
    """PV of one forward across ``n`` scenarios."""
    n = spot_path.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_fx_forward(
            spot_path[i], strike, expiry, rate_dom_path[i], rate_for_path[i], notional, direction,
        )
    return pv
```

### `pricing/kernels/rates.py`

```python
"""
Rates pricing kernels — par swaps + Bachelier (normal-vol) swaptions, ``@njit`` compiled.

Normal vols are used because swap rates can be negative. As with the FX kernels there are
scalar and ``*_vec`` (scenario-window) variants. Curve inputs are zero-rate pillars
``(curve_tenors, curve_values)``; the annuity and forward swap rate are rebuilt from them
so a shocked curve flows straight through into PnL.
"""
from __future__ import annotations

import math

import numpy as np
from src.rade_static_replication.pricing.kernels._numba import njit
from src.rade_static_replication.pricing.kernels._math import (
    norm_cdf,
    norm_pdf,
    compute_annuity,
    compute_forward_swap_rate,
)


@njit(cache=True)
def price_ir_swap(curve_tenors, curve_values, maturity, fixed_rate, notional, pay_receive_sign, freq):
    """PV of a par swap: ``sign * notional * (par_rate - fixed_rate) * annuity``."""
    annuity = compute_annuity(curve_tenors, curve_values, 0.0, maturity, freq)
    par_rate = compute_forward_swap_rate(curve_tenors, curve_values, 0.0, maturity, freq)
    return pay_receive_sign * notional * (par_rate - fixed_rate) * annuity


@njit(cache=True)
def price_ir_swaption_bachelier(
    curve_tenors, curve_values, option_expiry, swap_tenor, strike, notional, is_payer, normal_vol, freq,
):
    """Bachelier swaption PV on the forward swap rate."""
    annuity = compute_annuity(curve_tenors, curve_values, option_expiry, swap_tenor, freq)
    forward_rate = compute_forward_swap_rate(curve_tenors, curve_values, option_expiry, swap_tenor, freq)

    # Degenerate market: discounted intrinsic on the annuity.
    if normal_vol <= 0.0 or option_expiry <= 0.0:
        intrinsic = max(forward_rate - strike, 0.0) if is_payer else max(strike - forward_rate, 0.0)
        return notional * annuity * intrinsic

    sqrt_t = math.sqrt(option_expiry)
    standardised_moneyness = (forward_rate - strike) / (normal_vol * sqrt_t)
    payer_receiver_sign = 1.0 if is_payer else -1.0
    # Bachelier call/put value per unit annuity: sign*(F-K)*N(sign*d) + vol*sqrt(T)*phi(d).
    option_value = (
        payer_receiver_sign * (forward_rate - strike) * norm_cdf(payer_receiver_sign * standardised_moneyness)
        + normal_vol * sqrt_t * norm_pdf(standardised_moneyness)
    )
    return notional * annuity * option_value


@njit(cache=True)
def price_ir_swap_vec(curve_tenors, curve_paths, maturity, fixed_rate, notional, pay_receive_sign, freq):
    """PV of one swap across ``n`` scenarios; ``curve_paths`` is ``(n, n_pillars)``."""
    n = curve_paths.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_ir_swap(curve_tenors, curve_paths[i], maturity, fixed_rate, notional, pay_receive_sign, freq)
    return pv


@njit(cache=True)
def price_ir_swaption_bachelier_vec(
    curve_tenors, curve_paths, option_expiry, swap_tenor, strike, notional, is_payer, normal_vol_path, freq,
):
    """PV of one swaption across ``n`` scenarios; ``normal_vol_path`` is ``(n,)``."""
    n = curve_paths.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_ir_swaption_bachelier(
            curve_tenors, curve_paths[i], option_expiry, swap_tenor, strike,
            notional, is_payer, normal_vol_path[i], freq,
        )
    return pv
```

### `assets/__init__.py`

```python
"""
Asset-class plugins — the extension seam.

Each asset class is an isolated subpackage (``fx/``, ``rates/``) bundling a builder,
generator, and pricer behind the protocols in :mod:`assets.base`. :func:`default_registry`
wires the built-ins.
"""
from src.rade_static_replication.assets.base import (
    AssetClassPlugin,
    ElementaryGenerator,
    Pricer,
    Registry,
    RiskFactorBuilder,
)
from src.rade_static_replication.assets.registry import default_registry

__all__ = [
    "AssetClassPlugin", "Registry", "default_registry",
    "RiskFactorBuilder", "ElementaryGenerator", "Pricer",
]
```

### `assets/base.py`

```python
"""
Asset-class extension seam.

Everything that differs per asset class sits behind three protocols
(:class:`RiskFactorBuilder`, :class:`ElementaryGenerator`, :class:`Pricer`), bundled
into an :class:`AssetClassPlugin` and held in a :class:`Registry`. Adding an asset
class is: implement the three (one subpackage), bundle, register one line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

import numpy as np

from src.rade_static_replication.clients.base import MarketDataClient
from src.rade_static_replication.domain.contracts import RiskFactorData, RiskFactorSpec
from src.rade_static_replication.domain.errors import BuilderError
from src.rade_static_replication.domain.instruments import ElementaryTrade


class RiskFactorBuilder(Protocol):
    """Builds one factor's market data + shocks from the client."""

    def build(
        self, spec: RiskFactorSpec, dependencies: Dict[str, RiskFactorData],
        client: MarketDataClient, cob_date: str,
    ) -> RiskFactorData:
        ...


class ElementaryGenerator(Protocol):
    """Emits the elementary-trade grid for one factor."""

    def generate(self, rf: RiskFactorData, grid: Dict) -> List[ElementaryTrade]:
        ...


class Pricer(Protocol):
    """Prices one factor's elementary trades."""

    def base_prices(self, trades: List[ElementaryTrade], rf: RiskFactorData) -> Dict[str, float]:
        ...

    def scenario_pnl(
        self, trades: List[ElementaryTrade], rf: RiskFactorData, base: Dict[str, float],
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class AssetClassPlugin:
    """A complete asset-class plugin: builder + generator + pricer under one name."""
    name: str
    builder: RiskFactorBuilder
    generator: ElementaryGenerator
    pricer: Pricer


class Registry:
    """Maps an asset-class name to its :class:`AssetClassPlugin`."""

    def __init__(self) -> None:
        self._plugins: Dict[str, AssetClassPlugin] = {}

    def register(self, plugin: AssetClassPlugin) -> "Registry":
        self._plugins[plugin.name.lower()] = plugin
        return self

    def get(self, name: str) -> AssetClassPlugin:
        plugin = self._plugins.get(str(name).lower())
        if plugin is None:
            raise BuilderError(
                f"no asset-class plugin registered for {name!r}; registered: {sorted(self._plugins)}"
            )
        return plugin

    def __contains__(self, name: str) -> bool:
        return str(name).lower() in self._plugins

    @property
    def names(self) -> List[str]:
        return sorted(self._plugins)
```

### `assets/fx/__init__.py`

```python
"""
FX asset-class plugin.

Edit ``builder.py`` (market data), ``instruments.py``/``generator.py`` (replicating
basis), or ``pricer.py`` (models) in isolation. The bundle is exported as
:data:`FX_PLUGIN`.
"""
from src.rade_static_replication.assets.base import AssetClassPlugin
from src.rade_static_replication.assets.fx.builder import FXRiskFactorBuilder
from src.rade_static_replication.assets.fx.generator import FXElementaryGenerator
from src.rade_static_replication.assets.fx.pricer import FXPricer

FX_PLUGIN = AssetClassPlugin(
    name="fx",
    builder=FXRiskFactorBuilder(),
    generator=FXElementaryGenerator(),
    pricer=FXPricer(),
)

__all__ = ["FX_PLUGIN", "FXRiskFactorBuilder", "FXElementaryGenerator", "FXPricer"]
```

### `assets/fx/builder.py`

```python
"""FX risk-factor builder — assembles an FXSnapshot + FXScenarioSet for one pair."""
from __future__ import annotations

from typing import Dict

from src.rade_static_replication.assets.fx.instruments import cob_to_date, parse_fx_factor
from src.rade_static_replication.clients.base import MarketDataClient
from src.rade_static_replication.domain.contracts import RiskFactorData, RiskFactorSpec
from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.fx.instruments import Spot, VolSurface
from src.rade_static_replication.marketdata.fx.scenarios import FXScenarioSet
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot


class FXRiskFactorBuilder:
    """Turn raw client payloads into the typed FX market objects for one factor.

    A factor id like ``FX.SPOT.USD.EUR`` resolves to pair ``EURUSD`` with domestic
    (quote) ccy USD and foreign (base) ccy EUR; we fetch both discount curves, the spot,
    the vol surface, and the scenario shocks, then validate the scenario grids line up.
    """

    def build(
        self, spec: RiskFactorSpec, dependencies: Dict[str, RiskFactorData],
        client: MarketDataClient, cob_date: str,
    ) -> RiskFactorData:
        foreign_ccy, domestic_ccy, pair = parse_fx_factor(spec.factor_id)

        # --- COB base market ---
        spot = Spot(pair=pair, value=client.fx_spot(pair, cob_date))
        domestic_payload = client.discount_curve(domestic_ccy, cob_date)
        foreign_payload = client.discount_curve(foreign_ccy, cob_date)
        surface_payload = client.fx_vol_surface(pair, cob_date)

        snapshot = FXSnapshot(
            factor_id=spec.factor_id, asset_class="fx", as_of=cob_to_date(cob_date),
            spot=spot,
            domestic_curve=DiscountCurve(domestic_ccy, domestic_payload.tenors, domestic_payload.zero_rates),
            foreign_curve=DiscountCurve(foreign_ccy, foreign_payload.tenors, foreign_payload.zero_rates),
            vol_surface=VolSurface(
                surface_payload.expiries, surface_payload.strikes, surface_payload.vols,
                surface_payload.strike_convention, surface_payload.vol_type,
            ),
        )

        # --- shocked scenario states ---
        shocks = client.fx_shocks(pair, cob_date)
        scenarios = FXScenarioSet(
            factor_id=spec.factor_id, asset_class="fx", scenario_ids=shocks.scenario_ids,
            spot=shocks.spot, vol=shocks.vol, vol_expiries=shocks.vol_expiries, vol_strikes=shocks.vol_strikes,
            domestic_rate=shocks.domestic_rate, domestic_tenors=shocks.domestic_tenors,
            foreign_rate=shocks.foreign_rate, foreign_tenors=shocks.foreign_tenors,
        )
        scenarios.validate_against(snapshot)  # fail fast on any grid mismatch
        return RiskFactorData(spec=spec, snapshot=snapshot, scenarios=scenarios, dependencies=dependencies)
```

### `assets/fx/generator.py`

```python
"""FX elementary-trade generator — options/forwards on a forward × moneyness grid."""
from __future__ import annotations

from typing import Dict, List

from src.rade_static_replication.assets.fx.instruments import DEFAULT_GRID
from src.rade_static_replication.domain.contracts import RiskFactorData
from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot


class FXElementaryGenerator:
    """Place vanilla/digital options and forwards on a (forward × moneyness) grid.

    For each expiry we anchor strikes to the outright forward (``strike = moneyness ×
    forward``) so the grid is self-consistent across rate environments. A single
    ATM-forward is emitted per expiry.
    """

    def generate(self, rf: RiskFactorData, grid: Dict) -> List[ElementaryTrade]:
        snapshot: FXSnapshot = rf.snapshot  # type: ignore[assignment]
        moneyness_grid = grid.get("moneyness", DEFAULT_GRID["moneyness"])
        expiry_grid = grid.get("expiries", DEFAULT_GRID["expiries"])
        payoff_types = grid.get("instruments", DEFAULT_GRID["instruments"])
        factor_id = rf.factor_id

        trades: List[ElementaryTrade] = []
        for expiry in expiry_grid:
            forward = snapshot.forward(float(expiry))
            for moneyness in moneyness_grid:
                strike = forward * float(moneyness)
                params = {"strike": strike, "expiry": float(expiry), "moneyness": float(moneyness)}
                for payoff in payoff_types:
                    # The forward only makes sense at-the-money-forward; skip off-ATM forwards.
                    if payoff == "forward" and abs(float(moneyness) - 1.0) > 1e-9:
                        continue
                    trade_id = f"{factor_id}|{payoff.upper()}|K={strike:.6g}|T={float(expiry):.4g}"
                    trades.append(ElementaryTrade(
                        trade_id=trade_id, factor_id=factor_id, asset_class="fx",
                        payoff_type=payoff, parameters=dict(params), notional=1.0,
                    ))
        return trades
```

### `assets/fx/instruments.py`

```python
"""
FX elementary-instrument definitions and shared helpers.

Keeps the *what to emit* (grid defaults, payoff set) and small parsing helpers in one
place so the generator and pricer agree on conventions.
"""
from __future__ import annotations

import datetime as _dt
from typing import Tuple

DEFAULT_GRID = {
    "moneyness": [0.90, 0.95, 1.00, 1.05, 1.10],
    "expiries": [0.25, 0.50, 1.00],
    "instruments": ["call", "put", "digital_call", "forward"],
}


def parse_fx_factor(factor_id: str) -> Tuple[str, str, str]:
    """``FX.SPOT.USD.EUR`` -> (foreign='EUR', domestic='USD', pair='EURUSD')."""
    parts = factor_id.split(".")
    foreign, domestic = parts[-1], parts[-2]
    return foreign, domestic, f"{foreign}{domestic}"


def cob_to_date(cob: str) -> _dt.date:
    return _dt.datetime.strptime(str(cob), "%Y%m%d").date()
```

### `assets/fx/pricer.py`

```python
"""
FX pricer — Garman-Kohlhagen base PVs and vectorised scenario PnL.

Vols are read off the surface by moneyness (``strike / forward``) on the surface's own
strike axis. The scenario path interpolates rates and the vol plane *per expiry* (cached,
since many strikes share an expiry) and prices the whole window in compiled kernels.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.rade_static_replication.domain.contracts import RiskFactorData
from src.rade_static_replication.domain.errors import PricingError
from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.fx.scenarios import FXScenarioSet
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot
from src.rade_static_replication.pricing.kernels.fx import (
    price_fx_digital,
    price_fx_digital_vec,
    price_fx_forward,
    price_fx_forward_vec,
    price_fx_vanilla,
    price_fx_vanilla_vec,
)


def _interp_rate_path(rate_matrix: np.ndarray, tenors: np.ndarray, expiry: float) -> np.ndarray:
    """Interpolate a zero rate at ``expiry`` for every scenario row -> ``(n_scenarios,)``."""
    return np.array([np.interp(expiry, tenors, rate_matrix[i]) for i in range(rate_matrix.shape[0])])


def _vol_plane_at_expiry(vol_cube: np.ndarray, expiries: np.ndarray, expiry: float) -> np.ndarray:
    """Linearly interpolate the (scenario × strike) vol plane at a target ``expiry``."""
    if expiry <= expiries[0]:
        return vol_cube[:, 0, :]
    if expiry >= expiries[-1]:
        return vol_cube[:, -1, :]
    upper = int(np.searchsorted(expiries, expiry))
    lower_exp, upper_exp = expiries[upper - 1], expiries[upper]
    weight = (expiry - lower_exp) / (upper_exp - lower_exp)
    return (1.0 - weight) * vol_cube[:, upper - 1, :] + weight * vol_cube[:, upper, :]


class FXPricer:
    """Garman-Kohlhagen pricer for the FX elementary universe."""

    def base_prices(self, trades: List[ElementaryTrade], rf: RiskFactorData) -> Dict[str, float]:
        """COB present value of every elementary trade for this factor."""
        snapshot: FXSnapshot = rf.snapshot  # type: ignore[assignment]
        spot = snapshot.spot.value
        prices: Dict[str, float] = {}
        for trade in trades:
            expiry, strike = trade.parameters["expiry"], trade.parameters["strike"]
            rate_dom = snapshot.domestic_curve.zero(expiry)
            rate_for = snapshot.foreign_curve.zero(expiry)
            # Surface is keyed by moneyness = strike / forward.
            vol = snapshot.vol_surface.vol(strike / snapshot.forward(expiry), expiry)
            prices[trade.trade_id] = self._price_scalar(
                trade.payoff_type, spot, strike, expiry, rate_dom, rate_for, vol, trade.notional,
            )
        return prices

    def scenario_pnl(
        self, trades: List[ElementaryTrade], rf: RiskFactorData, base: Dict[str, float],
    ) -> np.ndarray:
        """Scenario PnL matrix ``(n_trades, n_scenarios)`` = PV(scenario) - PV(base)."""
        scenarios: FXScenarioSet = rf.scenarios  # type: ignore[assignment]
        n_scenarios = scenarios.n_scenarios
        spot_path = scenarios.spot
        strike_axis = scenarios.vol_strikes

        # Market quantities depend only on expiry, so cache them across trades.
        market_by_expiry: Dict[float, dict] = {}

        def _market_at(expiry: float) -> dict:
            key = round(expiry, 8)
            if key not in market_by_expiry:
                rate_dom = _interp_rate_path(scenarios.domestic_rate, scenarios.domestic_tenors, expiry)
                rate_for = _interp_rate_path(scenarios.foreign_rate, scenarios.foreign_tenors, expiry)
                market_by_expiry[key] = {
                    "rate_dom": rate_dom,
                    "rate_for": rate_for,
                    "forward": spot_path * np.exp((rate_dom - rate_for) * expiry),
                    "vol_plane": _vol_plane_at_expiry(scenarios.vol, scenarios.vol_expiries, expiry),
                }
            return market_by_expiry[key]

        pnl = np.empty((len(trades), n_scenarios), dtype=np.float64)
        for row, trade in enumerate(trades):
            expiry, strike = trade.parameters["expiry"], trade.parameters["strike"]
            market = _market_at(expiry)
            moneyness = strike / market["forward"]
            # Per-scenario vol: interpolate along the strike axis at each scenario's moneyness.
            vol_path = np.array([
                np.interp(moneyness[i], strike_axis, market["vol_plane"][i]) for i in range(n_scenarios)
            ])
            scenario_pv = self._price_vector(
                trade.payoff_type, spot_path, strike, expiry,
                market["rate_dom"], market["rate_for"], vol_path, trade.notional,
            )
            pnl[row] = scenario_pv - base[trade.trade_id]
        return pnl

    # ---- payoff routing ----

    @staticmethod
    def _price_scalar(payoff, spot, strike, expiry, rate_dom, rate_for, vol, notional) -> float:
        if payoff == "call":
            return price_fx_vanilla(spot, strike, expiry, rate_dom, rate_for, vol, True, notional)
        if payoff == "put":
            return price_fx_vanilla(spot, strike, expiry, rate_dom, rate_for, vol, False, notional)
        if payoff == "digital_call":
            return price_fx_digital(spot, strike, expiry, rate_dom, rate_for, vol, True, notional)
        if payoff == "digital_put":
            return price_fx_digital(spot, strike, expiry, rate_dom, rate_for, vol, False, notional)
        if payoff == "forward":
            return price_fx_forward(spot, strike, expiry, rate_dom, rate_for, notional, 1.0)
        raise PricingError(f"unknown FX payoff {payoff!r}")

    @staticmethod
    def _price_vector(payoff, spot_path, strike, expiry, rate_dom, rate_for, vol_path, notional) -> np.ndarray:
        if payoff == "call":
            return price_fx_vanilla_vec(spot_path, strike, expiry, rate_dom, rate_for, vol_path, True, notional)
        if payoff == "put":
            return price_fx_vanilla_vec(spot_path, strike, expiry, rate_dom, rate_for, vol_path, False, notional)
        if payoff == "digital_call":
            return price_fx_digital_vec(spot_path, strike, expiry, rate_dom, rate_for, vol_path, True, notional)
        if payoff == "digital_put":
            return price_fx_digital_vec(spot_path, strike, expiry, rate_dom, rate_for, vol_path, False, notional)
        if payoff == "forward":
            return price_fx_forward_vec(spot_path, strike, expiry, rate_dom, rate_for, notional, 1.0)
        raise PricingError(f"unknown FX payoff {payoff!r}")
```

### `assets/rates/__init__.py`

```python
"""
Rates asset-class plugin.

Edit ``builder.py`` (curve/cube), ``instruments.py``/``generator.py`` (swaption grid),
or ``pricer.py`` (Bachelier) in isolation. Exported as :data:`RATES_PLUGIN`.
"""
from src.rade_static_replication.assets.base import AssetClassPlugin
from src.rade_static_replication.assets.rates.builder import RatesRiskFactorBuilder
from src.rade_static_replication.assets.rates.generator import RatesElementaryGenerator
from src.rade_static_replication.assets.rates.pricer import RatesPricer

RATES_PLUGIN = AssetClassPlugin(
    name="rates",
    builder=RatesRiskFactorBuilder(),
    generator=RatesElementaryGenerator(),
    pricer=RatesPricer(),
)

__all__ = ["RATES_PLUGIN", "RatesRiskFactorBuilder", "RatesElementaryGenerator", "RatesPricer"]
```

### `assets/rates/builder.py`

```python
"""Rates risk-factor builder — assembles a RatesSnapshot + RatesScenarioSet."""
from __future__ import annotations

from typing import Dict

from src.rade_static_replication.assets.rates.instruments import cob_to_date, parse_rates_factor
from src.rade_static_replication.clients.base import MarketDataClient
from src.rade_static_replication.domain.contracts import RiskFactorData, RiskFactorSpec
from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.rates.instruments import VolCube
from src.rade_static_replication.marketdata.rates.scenarios import RatesScenarioSet
from src.rade_static_replication.marketdata.rates.snapshot import RatesSnapshot


class RatesRiskFactorBuilder:
    """Turn raw client payloads into the typed rates market objects for one factor.

    The currency is the last token of the factor id (e.g. ``IR_CURVE_SWAP.GBP`` -> GBP).
    We fetch the discount curve, the optional swaption cube, and the scenario shocks.
    """

    def build(
        self, spec: RiskFactorSpec, dependencies: Dict[str, RiskFactorData],
        client: MarketDataClient, cob_date: str,
    ) -> RiskFactorData:
        currency = parse_rates_factor(spec.factor_id)

        # --- COB base market ---
        curve_payload = client.discount_curve(currency, cob_date)
        cube_payload = client.rates_vol_cube(spec.factor_id, cob_date)  # may be None
        snapshot = RatesSnapshot(
            factor_id=spec.factor_id, asset_class="rates", as_of=cob_to_date(cob_date),
            discount_curve=DiscountCurve(currency, curve_payload.tenors, curve_payload.zero_rates),
            vol_cube=(
                None if cube_payload is None else
                VolCube(
                    cube_payload.expiries, cube_payload.swap_tenors,
                    cube_payload.strikes, cube_payload.vols, cube_payload.vol_type,
                )
            ),
        )

        # --- shocked scenario states ---
        shocks = client.rates_shocks(spec.factor_id, cob_date)
        scenarios = RatesScenarioSet(
            factor_id=spec.factor_id, asset_class="rates", scenario_ids=shocks.scenario_ids,
            curve=shocks.curve, curve_tenors=shocks.curve_tenors,
            vol=shocks.vol, vol_expiries=shocks.vol_expiries,
            vol_swap_tenors=shocks.vol_swap_tenors, vol_strikes=shocks.vol_strikes,
        )
        scenarios.validate_against(snapshot)  # fail fast on any grid mismatch
        return RiskFactorData(spec=spec, snapshot=snapshot, scenarios=scenarios, dependencies=dependencies)
```

### `assets/rates/generator.py`

```python
"""Rates elementary-trade generator — payer/receiver swaptions on an expiry × tenor × strike grid."""
from __future__ import annotations

from typing import Dict, List

from src.rade_static_replication.assets.rates.instruments import DEFAULT_GRID
from src.rade_static_replication.domain.contracts import RiskFactorData
from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.rates.snapshot import RatesSnapshot
from src.rade_static_replication.pricing.kernels._math import compute_forward_swap_rate


class RatesElementaryGenerator:
    """Emit swaptions around the ATM forward swap rate for each (expiry, swap-tenor).

    Strikes are placed at basis-point offsets from the ATM forward swap rate so the grid
    straddles the money for every (expiry, tenor) point regardless of curve level.
    """

    def generate(self, rf: RiskFactorData, grid: Dict) -> List[ElementaryTrade]:
        snapshot: RatesSnapshot = rf.snapshot  # type: ignore[assignment]
        curve_tenors = snapshot.discount_curve.tenors
        curve_zeros = snapshot.discount_curve.zero_rates
        expiry_grid = grid.get("expiries", DEFAULT_GRID["expiries"])
        swap_tenor_grid = grid.get("swap_tenors", DEFAULT_GRID["swap_tenors"])
        strike_offsets_bp = grid.get("strike_offsets_bp", DEFAULT_GRID["strike_offsets_bp"])
        payoff_types = grid.get("instruments", DEFAULT_GRID["instruments"])
        pay_frequency = grid.get("freq", DEFAULT_GRID["freq"])
        factor_id = rf.factor_id

        trades: List[ElementaryTrade] = []
        for expiry in expiry_grid:
            for swap_tenor in swap_tenor_grid:
                atm_rate = compute_forward_swap_rate(
                    curve_tenors, curve_zeros, float(expiry), float(swap_tenor), float(pay_frequency),
                )
                for offset_bp in strike_offsets_bp:
                    strike = atm_rate + float(offset_bp) * 1e-4  # bp -> absolute rate
                    for payoff in payoff_types:
                        trade_id = (
                            f"{factor_id}|{payoff.upper()}"
                            f"|E={float(expiry):.4g}|Tn={float(swap_tenor):.4g}|K={strike:.6g}"
                        )
                        trades.append(ElementaryTrade(
                            trade_id=trade_id, factor_id=factor_id, asset_class="rates", payoff_type=payoff,
                            parameters={
                                "expiry": float(expiry), "swap_tenor": float(swap_tenor),
                                "strike": strike, "freq": float(pay_frequency),
                            },
                            notional=1.0,
                        ))
        return trades
```

### `assets/rates/instruments.py`

```python
"""Rates elementary-instrument definitions and helpers."""
from __future__ import annotations

import datetime as _dt

DEFAULT_GRID = {
    "expiries": [1.0, 2.0, 5.0],
    "swap_tenors": [2.0, 5.0, 10.0],
    "strike_offsets_bp": [-50.0, 0.0, 50.0],
    "instruments": ["payer", "receiver"],
    "freq": 0.5,
}


def parse_rates_factor(factor_id: str) -> str:
    """Return the currency token, e.g. ``IR_CURVE_SWAP.GBP`` -> ``GBP``."""
    return factor_id.split(".")[-1]


def cob_to_date(cob: str) -> _dt.date:
    return _dt.datetime.strptime(str(cob), "%Y%m%d").date()
```

### `assets/rates/pricer.py`

```python
"""
Rates pricer — Bachelier swaptions, base PV + vectorised scenario PnL.

Scenario PnL reprices each swaption across the shocked curve matrix; normal vols come
from the scenario cube (trilinear, vectorised over scenarios) when present, otherwise the
base cube vol is held flat across the window.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.rade_static_replication.domain.contracts import RiskFactorData
from src.rade_static_replication.domain.errors import PricingError
from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.rates.scenarios import RatesScenarioSet
from src.rade_static_replication.marketdata.rates.snapshot import RatesSnapshot
from src.rade_static_replication.pricing.kernels.rates import (
    price_ir_swaption_bachelier,
    price_ir_swaption_bachelier_vec,
)

# Used only when no swaption cube is supplied (≈ 80bp normal vol).
_FALLBACK_NORMAL_VOL = 0.008


def _bracket(axis: np.ndarray, value: float) -> Tuple[int, int, float]:
    """Return ``(lower_idx, upper_idx, weight)`` to linearly interpolate ``value`` on ``axis``."""
    if value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        return axis.size - 1, axis.size - 1, 0.0
    upper = int(np.searchsorted(axis, value))
    lower = upper - 1
    weight = (value - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, weight


def _interp_cube_vol_path(
    vol_cube_path: np.ndarray, expiries, swap_tenors, strikes, expiry, swap_tenor, strike,
) -> np.ndarray:
    """Trilinear vol at one (expiry, tenor, strike) for every scenario -> ``(n_scenarios,)``.

    ``vol_cube_path`` is ``(n_scenarios, n_exp, n_ten, n_k)``; we collapse the last three
    axes with the bracketing weights, keeping the scenario axis vectorised.
    """
    e_lo, e_hi, e_w = _bracket(expiries, expiry)
    t_lo, t_hi, t_w = _bracket(swap_tenors, swap_tenor)
    k_lo, k_hi, k_w = _bracket(strikes, strike)
    vol = (1 - e_w) * vol_cube_path[:, e_lo] + e_w * vol_cube_path[:, e_hi]   # -> (n, n_ten, n_k)
    vol = (1 - t_w) * vol[:, t_lo] + t_w * vol[:, t_hi]                       # -> (n, n_k)
    return (1 - k_w) * vol[:, k_lo] + k_w * vol[:, k_hi]                      # -> (n,)


class RatesPricer:
    """Bachelier pricer for the rates (swaption) elementary universe."""

    def base_prices(self, trades: List[ElementaryTrade], rf: RiskFactorData) -> Dict[str, float]:
        """COB present value of every swaption for this factor."""
        snapshot: RatesSnapshot = rf.snapshot  # type: ignore[assignment]
        tenors, zero_rates = snapshot.discount_curve.tenors, snapshot.discount_curve.zero_rates
        prices: Dict[str, float] = {}
        for trade in trades:
            params = trade.parameters
            normal_vol = (
                snapshot.vol_cube.vol(params["expiry"], params["swap_tenor"], params["strike"])
                if snapshot.vol_cube is not None else _FALLBACK_NORMAL_VOL
            )
            prices[trade.trade_id] = price_ir_swaption_bachelier(
                tenors, zero_rates, params["expiry"], params["swap_tenor"], params["strike"],
                trade.notional, self._is_payer(trade.payoff_type), normal_vol, params["freq"],
            )
        return prices

    def scenario_pnl(
        self, trades: List[ElementaryTrade], rf: RiskFactorData, base: Dict[str, float],
    ) -> np.ndarray:
        """Scenario PnL matrix ``(n_trades, n_scenarios)`` = PV(scenario) - PV(base)."""
        snapshot: RatesSnapshot = rf.snapshot  # type: ignore[assignment]
        scenarios: RatesScenarioSet = rf.scenarios  # type: ignore[assignment]
        n_scenarios = scenarios.n_scenarios
        curve_tenors = scenarios.curve_tenors
        curve_paths = scenarios.curve

        pnl = np.empty((len(trades), n_scenarios), dtype=np.float64)
        for row, trade in enumerate(trades):
            params = trade.parameters
            if scenarios.vol is not None:
                normal_vol_path = _interp_cube_vol_path(
                    scenarios.vol, scenarios.vol_expiries, scenarios.vol_swap_tenors, scenarios.vol_strikes,
                    params["expiry"], params["swap_tenor"], params["strike"],
                )
            else:
                base_vol = (
                    snapshot.vol_cube.vol(params["expiry"], params["swap_tenor"], params["strike"])
                    if snapshot.vol_cube is not None else _FALLBACK_NORMAL_VOL
                )
                normal_vol_path = np.full(n_scenarios, base_vol)

            scenario_pv = price_ir_swaption_bachelier_vec(
                curve_tenors, curve_paths, params["expiry"], params["swap_tenor"], params["strike"],
                trade.notional, self._is_payer(trade.payoff_type), normal_vol_path, params["freq"],
            )
            pnl[row] = scenario_pv - base[trade.trade_id]
        return pnl

    @staticmethod
    def _is_payer(payoff: str) -> bool:
        if payoff == "payer":
            return True
        if payoff == "receiver":
            return False
        raise PricingError(f"unknown rates payoff {payoff!r}")
```

### `assets/registry.py`

```python
"""
Default plugin registry.

``default_registry()`` returns a :class:`Registry` pre-loaded with the built-in FX and
Rates plugins. To add an asset class: build its subpackage under ``assets/<class>/``
and register one line here (or call ``registry.register(...)`` at the call site).
"""
from __future__ import annotations

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.assets.fx import FX_PLUGIN
from src.rade_static_replication.assets.rates import RATES_PLUGIN


def default_registry() -> Registry:
    """A registry with the built-in asset classes installed."""
    return Registry().register(FX_PLUGIN).register(RATES_PLUGIN)
```

### `portfolio/__init__.py`

```python
"""Portfolio domain logic: normalisation, validation, risk-factor resolution."""
from src.rade_static_replication.portfolio.normalise import normalise
from src.rade_static_replication.portfolio.resolution import resolve
from src.rade_static_replication.portfolio.validate import validate

__all__ = ["normalise", "validate", "resolve"]
```

### `portfolio/normalise.py`

```python
"""
Portfolio normalisation (stage 2).

Collapses the exploded raw export (one row per trade × risk-type × curve, notional
row tagged ``AssetClass="ALL"``) into one validated row per trade:

1. pivot ``RiskType`` into columns;
2. resolve ``AssetClass`` ignoring the ``"ALL"`` notional rows;
3. carry through every column constant within a trade (discovered dynamically);
4. derive ``NotionalSign``, ``SignedNotional``, ``yrs_to_maturity``.

Raw column names live in one place (below) — point them at your export if it differs.
"""
from __future__ import annotations

import datetime as _dt
import logging

import numpy as np
import pandas as pd

from src.rade_static_replication.domain.contracts import Portfolio, RawPortfolio
from src.rade_static_replication.domain.errors import PortfolioError

logger = logging.getLogger(__name__)

TRADE_ID_COL = "PTSDealNumber"
RISK_TYPE_COL = "RiskType"
RISK_VALUE_COL = "Sum_RiskValuesUSD_Net"
ASSET_CLASS_COL = "AssetClass"
BUY_SELL_COL = "BuySellInd"
MATURITY_COL = "MaturityDate"
NOTIONAL_RISK_TYPE = "Notional"
ALL_SENTINEL = "ALL"

# Per-(trade, curve) noise that should not be carried as attributes.
_DROP_COLS = {"PTSCurveCode", "StandardCurveCode"}


def _years_to_maturity(maturity: str, cob_date: str) -> float:
    try:
        m = _dt.datetime.strptime(str(maturity), "%Y%m%d").date()
        c = _dt.datetime.strptime(str(cob_date), "%Y%m%d").date()
        return max((m - c).days / 365.0, 0.0)
    except (ValueError, TypeError):
        return np.nan


def _first_non_null(s: pd.Series):
    for v in s:
        if pd.notna(v):
            return v
    return np.nan


def _first_non_all(s: pd.Series):
    for v in s:
        if pd.notna(v) and str(v) != ALL_SENTINEL:
            return v
    return np.nan


def normalise(raw: RawPortfolio) -> Portfolio:
    """Collapse the exploded raw export into one row per trade."""
    attributes_raw = raw.attributes.copy()
    if TRADE_ID_COL not in attributes_raw.columns:
        raise PortfolioError(f"raw attributes missing trade-id column {TRADE_ID_COL!r}")
    attributes_raw[TRADE_ID_COL] = attributes_raw[TRADE_ID_COL].astype(str)

    # 1) Pivot the risk-type rows into one column per risk type (Notional, FXPV, ...).
    risk_values = attributes_raw.pivot_table(
        index=TRADE_ID_COL, columns=RISK_TYPE_COL, values=RISK_VALUE_COL, aggfunc="first",
    )
    risk_values.columns = [str(c) for c in risk_values.columns]

    # 2) Carry through every other column that is constant within a trade. AssetClass uses
    #    a special reducer that ignores the "ALL" notional-row sentinel.
    exclude = {RISK_TYPE_COL, RISK_VALUE_COL, TRADE_ID_COL} | _DROP_COLS
    carry_columns = [c for c in attributes_raw.columns if c not in exclude]
    by_trade = attributes_raw.groupby(TRADE_ID_COL, sort=False)

    constant_attributes = pd.DataFrame(index=risk_values.index)
    for column in carry_columns:
        reducer = _first_non_all if column == ASSET_CLASS_COL else _first_non_null
        constant_attributes[column] = by_trade[column].apply(reducer)

    attributes = constant_attributes.join(risk_values)

    # 3) Derive direction and signed notional from the buy/sell indicator.
    notional_sign = attributes[BUY_SELL_COL].map(
        lambda flag: 1.0 if str(flag).upper().startswith("B") else -1.0
    )
    attributes["NotionalSign"] = notional_sign
    if NOTIONAL_RISK_TYPE in attributes.columns:
        attributes["SignedNotional"] = notional_sign * attributes[NOTIONAL_RISK_TYPE].astype(float)
    if MATURITY_COL in attributes.columns:
        attributes["yrs_to_maturity"] = attributes[MATURITY_COL].map(
            lambda maturity: _years_to_maturity(maturity, raw.cob_date)
        )

    attributes.index = attributes.index.astype(str)
    attributes.index.name = "trade_id"

    # 4) Align the target PnL frame to a clean trade_id index and keep only scenario columns.
    target = raw.target_pnl.copy()
    if target.index.name != "trade_id" and "trade_id" in target.columns:
        target = target.set_index("trade_id")
    target.index = target.index.astype(str)
    scenario_cols = [c for c in target.columns if c != ASSET_CLASS_COL]
    target = target[scenario_cols]

    portfolio = Portfolio(
        attributes=attributes, target_pnl=target,
        scenario_ids=np.array(scenario_cols), cob_date=raw.cob_date,
    )
    logger.info(
        "Normalised portfolio: %d trades, %d asset classes, %d scenarios",
        len(attributes), len(portfolio.asset_classes), len(scenario_cols),
    )
    return portfolio
```

### `portfolio/resolution/__init__.py`

```python
"""Config-driven risk-factor resolution."""
from src.rade_static_replication.portfolio.resolution.resolver import resolve

__all__ = ["resolve"]
```

### `portfolio/resolution/resolver.py`

```python
"""
Risk-factor resolution (stage 3).

Walks the portfolio, applies each asset class's :class:`AssetFactorRule`, and returns
a de-duplicated :class:`RiskFactorUniverse`: every primary factor, every dependency
factor needed to build them, and the trade -> primary-factors map used downstream.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import (
    Portfolio,
    RiskFactorSpec,
    RiskFactorUniverse,
)
from src.rade_static_replication.domain.errors import FactorResolutionError
from src.rade_static_replication.portfolio.resolution.rules import MappingCache, resolve_row

logger = logging.getLogger(__name__)


def resolve(portfolio: Portfolio, config: OrchestratorConfig) -> RiskFactorUniverse:
    """Resolve the full risk-factor universe for the portfolio."""
    rules = {k.lower(): v for k, v in config.factors.rules.items()}
    if not rules:
        raise FactorResolutionError("no factor-resolution rules configured")

    specs: Dict[str, RiskFactorSpec] = {}
    factors_by_trade: Dict[str, List[str]] = {}
    mappings = MappingCache()

    for trade_id, row in portfolio.attributes.iterrows():
        asset_class = str(row.get("AssetClass", "")).lower()
        rule = rules.get(asset_class)
        if rule is None:
            logger.warning("no resolution rule for asset class %r (trade %s); skipped", asset_class, trade_id)
            factors_by_trade[str(trade_id)] = []
            continue

        primaries, dependencies = resolve_row(rule, row, mappings)
        dep_ids = tuple(d.factor_id for d in dependencies)

        for dep in dependencies:
            specs.setdefault(dep.factor_id, dep)

        primary_ids: List[str] = []
        for prim in primaries:
            spec = RiskFactorSpec(
                factor_id=prim.factor_id, asset_class=prim.asset_class,
                dependencies=dep_ids, is_primary=True, meta=prim.meta,
            )
            existing = specs.get(prim.factor_id)
            if existing is None or not existing.is_primary:
                specs[prim.factor_id] = spec
            primary_ids.append(prim.factor_id)

        factors_by_trade[str(trade_id)] = primary_ids

    if not specs:
        raise FactorResolutionError("resolution produced no risk factors")

    universe = RiskFactorUniverse(specs=specs, factors_by_trade=factors_by_trade)
    logger.info(
        "Resolved %d risk factors (%d primary, %d dependency-only)",
        len(specs), len(universe.primary_ids), len(specs) - len(universe.primary_ids),
    )
    return universe
```

### `portfolio/resolution/rules.py`

```python
"""
Rule application helpers for risk-factor resolution.

Pure functions that turn one trade row + an :class:`AssetFactorRule` into primary and
dependency :class:`RiskFactorSpec`s. Mapping CSVs are cached. The trivial FX
numeraire identity (``FX.SPOT.USD.USD``) is dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.rade_static_replication.config.schema import AssetFactorRule
from src.rade_static_replication.domain.contracts import RiskFactorSpec
from src.rade_static_replication.domain.errors import FactorResolutionError


def is_fx_identity(factor_id: str) -> bool:
    """True for the trivial USD-per-USD FX leg."""
    return factor_id.upper().endswith(".USD.USD")


def _valid(value) -> bool:
    return value is not None and pd.notna(value) and str(value).strip() != ""


class MappingCache:
    """Lazily loads and indexes mapping CSVs, keyed by path."""

    def __init__(self) -> None:
        self._cache: Dict[str, pd.DataFrame] = {}

    def get(self, path: str, key_field: str) -> pd.DataFrame:
        if path not in self._cache:
            df = pd.read_csv(Path(path))
            key_col = key_field if key_field in df.columns else df.columns[0]
            self._cache[path] = df.set_index(df[key_col].astype(str))
        return self._cache[path]


def resolve_row(
    rule: AssetFactorRule, row: pd.Series, mappings: MappingCache,
) -> Tuple[List[RiskFactorSpec], List[RiskFactorSpec]]:
    """Resolve one trade row into (primary specs, dependency specs)."""
    if rule.source == "mapping":
        key = row.get(rule.key_field)
        if not _valid(key):
            raise FactorResolutionError(f"[{rule.asset_class}] trade missing mapping key {rule.key_field!r}")
        table = mappings.get(rule.mapping_path, rule.key_field)
        if str(key) not in table.index:
            raise FactorResolutionError(
                f"[{rule.asset_class}] key {key!r} not found in mapping {rule.mapping_path}"
            )
        lookup = table.loc[str(key)]
        source_row = {f: lookup.get(f) for f in (rule.factor_fields + rule.dependency_fields)}
        meta_key = {"key": str(key)}
    else:
        source_row = {f: row.get(f) for f in (rule.factor_fields + rule.dependency_fields)}
        meta_key = {}

    primaries = _make_specs(rule, rule.factor_fields, source_row, is_primary=True, meta_key=meta_key)
    dependencies = _make_specs(rule, rule.dependency_fields, source_row, is_primary=False, meta_key=meta_key)
    return primaries, dependencies


def _make_specs(
    rule: AssetFactorRule, fields: List[str], source_row: Dict, *, is_primary: bool, meta_key: Dict,
) -> List[RiskFactorSpec]:
    out: List[RiskFactorSpec] = []
    seen: set[str] = set()
    for field_name in fields:
        factor_id = source_row.get(field_name)
        if not _valid(factor_id):
            continue
        factor_id = str(factor_id).strip()
        if is_fx_identity(factor_id) or factor_id in seen:
            continue
        seen.add(factor_id)
        out.append(RiskFactorSpec(
            factor_id=factor_id,
            asset_class=rule.asset_class_of.get(field_name, rule.asset_class),
            is_primary=is_primary,
            meta={"field": field_name, **meta_key},
        ))
    return out
```

### `portfolio/validate.py`

```python
"""
Portfolio validation (stage 2, second half).

Fail-fast on the invariants the rest of the pipeline assumes; return non-fatal
warnings for the caller/log.
"""
from __future__ import annotations

import logging
from typing import List

from src.rade_static_replication.domain.contracts import Portfolio
from src.rade_static_replication.domain.errors import PortfolioError
from src.rade_static_replication.portfolio.normalise import ASSET_CLASS_COL

logger = logging.getLogger(__name__)


def validate(portfolio: Portfolio) -> List[str]:
    """Raise :class:`PortfolioError` on any breach; return warnings list."""
    attr = portfolio.attributes
    warnings: List[str] = []

    if attr.empty:
        raise PortfolioError("portfolio has no trades after normalisation")
    if not attr.index.is_unique:
        dupes = attr.index[attr.index.duplicated()].unique().tolist()
        raise PortfolioError(f"duplicate trade ids after normalisation: {dupes[:5]}")
    if ASSET_CLASS_COL not in attr.columns:
        raise PortfolioError(f"normalised attributes missing {ASSET_CLASS_COL!r}")
    missing_ac = attr.index[attr[ASSET_CLASS_COL].isna()].tolist()
    if missing_ac:
        raise PortfolioError(f"{len(missing_ac)} trades have no resolved AssetClass: {missing_ac[:5]}")

    missing_pnl = [t for t in portfolio.trade_ids if t not in portfolio.target_pnl.index]
    if missing_pnl:
        warnings.append(f"{len(missing_pnl)} trades have no target PnL row")
    if portfolio.scenario_ids.size == 0:
        raise PortfolioError("target PnL has no scenario columns")

    for w in warnings:
        logger.warning("portfolio validation: %s", w)
    return warnings
```

### `config/__init__.py`

```python
"""Configuration: the OrchestratorConfig schema and its YAML loader."""
from src.rade_static_replication.config.loader import config_from_dict, load_config
from src.rade_static_replication.config.schema import (
    AssetFactorRule,
    ClusteringConfig,
    ElementaryConfig,
    EngineConfig,
    FactorResolutionConfig,
    OrchestratorConfig,
    OutputConfig,
)

__all__ = [
    "OrchestratorConfig", "AssetFactorRule", "FactorResolutionConfig",
    "ElementaryConfig", "EngineConfig", "ClusteringConfig", "OutputConfig",
    "load_config", "config_from_dict",
]
```

### `config/loader.py`

```python
"""
Configuration loading — YAML/dict -> validated :class:`OrchestratorConfig`.

Relative ``mapping_path`` entries resolve against the YAML file's directory so a
config is portable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.rade_static_replication.config.schema import (
    AssetFactorRule,
    ClusteringConfig,
    ElementaryConfig,
    EngineConfig,
    FactorResolutionConfig,
    OrchestratorConfig,
    OutputConfig,
)
from src.rade_static_replication.domain.errors import ConfigurationError


def load_config(path: str | Path) -> OrchestratorConfig:
    """Load and validate an :class:`OrchestratorConfig` from a YAML file."""
    import yaml

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config file not found: {path}") from exc
    return config_from_dict(raw, base_dir=path.parent)


def config_from_dict(raw: Dict[str, Any], base_dir: Optional[Path] = None) -> OrchestratorConfig:
    """Build a config from an already-parsed dict."""
    if "cob_date" not in raw:
        raise ConfigurationError("config missing required 'cob_date'")

    rules: Dict[str, AssetFactorRule] = {}
    for ac, block in (raw.get("factors", {}) or {}).items():
        mp = block.get("mapping_path")
        if mp and base_dir is not None and not Path(mp).is_absolute():
            mp = str((base_dir / mp).resolve())
        rules[ac] = AssetFactorRule(
            asset_class=ac,
            source=block["source"],
            key_field=block.get("key_field", ""),
            factor_fields=list(block.get("factor_fields", [])),
            dependency_fields=list(block.get("dependency_fields", [])),
            asset_class_of=dict(block.get("asset_class_of", {})),
            mapping_path=mp,
        )

    out_block = raw.get("output", {}) or {}
    eng = raw.get("engine", {}) or {}
    clus = raw.get("clustering", {}) or {}
    return OrchestratorConfig(
        cob_date=str(raw["cob_date"]),
        factors=FactorResolutionConfig(rules=rules),
        elementary=ElementaryConfig(grids=dict(raw.get("elementary", {}) or {})),
        engine=EngineConfig(
            max_workers=int(eng.get("max_workers", 1)),
            use_numba=bool(eng.get("use_numba", True)),
        ),
        clustering=ClusteringConfig(keys=list(clus.get("keys", ["AssetClass"]))),
        output=OutputConfig(
            root=Path(out_block.get("root", "data/rade_static_replication")),
            run_id=out_block.get("run_id", "run"),
        ),
    )
```

### `config/schema.py`

```python
"""
Configuration schema — pure, validated dataclasses.

The structure mirrors the pipeline stages so a reader maps config block -> stage at
a glance. Parsing lives in :mod:`config.loader`; this module is data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rade_static_replication.domain.errors import ConfigurationError


@dataclass(frozen=True)
class AssetFactorRule:
    """How to resolve risk factors for one asset class.

    ``source`` is ``"attribute"`` (factor ids already on the trade row) or
    ``"mapping"`` (look a ``key_field`` up in a CSV). ``asset_class_of`` maps each
    factor field to the asset class of the factor it yields, so one rule can emit
    factors of several classes (FX needs IR dependency curves).
    """
    asset_class: str
    source: str
    key_field: str
    factor_fields: List[str]
    dependency_fields: List[str] = field(default_factory=list)
    asset_class_of: Dict[str, str] = field(default_factory=dict)
    mapping_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.source not in ("attribute", "mapping"):
            raise ConfigurationError(
                f"[{self.asset_class}] source must be 'attribute' or 'mapping', got {self.source!r}"
            )
        if self.source == "mapping" and not self.mapping_path:
            raise ConfigurationError(f"[{self.asset_class}] mapping source needs 'mapping_path'")
        if not self.factor_fields:
            raise ConfigurationError(f"[{self.asset_class}] needs at least one factor_field")


@dataclass(frozen=True)
class FactorResolutionConfig:
    rules: Dict[str, AssetFactorRule] = field(default_factory=dict)


@dataclass(frozen=True)
class ElementaryConfig:
    """Per-asset-class elementary-grid parameters (opaque dict per class)."""
    grids: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineConfig:
    max_workers: int = 1
    use_numba: bool = True


@dataclass(frozen=True)
class ClusteringConfig:
    """Attribute columns whose value-tuples define clusters (AssetClass implied first)."""
    keys: List[str] = field(default_factory=lambda: ["AssetClass"])


@dataclass(frozen=True)
class OutputConfig:
    """Artifact-store location."""
    root: Path = Path("data/rade_static_replication")
    run_id: str = "run"


@dataclass(frozen=True)
class OrchestratorConfig:
    """Top-level run recipe (one per invocation)."""
    cob_date: str
    factors: FactorResolutionConfig = field(default_factory=FactorResolutionConfig)
    elementary: ElementaryConfig = field(default_factory=ElementaryConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
```

### `pipeline/__init__.py`

```python
"""Pipeline orchestration: the RunContext, the stage functions, and the Orchestrator."""
from src.rade_static_replication.pipeline.context import RunContext
from src.rade_static_replication.pipeline.orchestrator import Orchestrator

__all__ = ["Orchestrator", "RunContext"]
```

### `pipeline/context.py`

```python
"""
Run context — the orchestrator's working ledger.

A mutable record of one run: the config, the clients/registry, and each stage's output
contract. Every stage reads the contracts it needs and writes exactly one. Keeping
state here (rather than threading many returns) makes partial runs, inspection, and
re-slicing trivial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.clients.base import MarketDataClient, PortfolioClient
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import (
    BasePriceSet,
    ClusterSet,
    ElementaryUniverse,
    FactorDataSet,
    PnLResult,
    Portfolio,
    RawPortfolio,
    RiskFactorUniverse,
)


@dataclass
class RunContext:
    """Mutable per-run state shared across stages."""
    config: OrchestratorConfig
    portfolio_client: PortfolioClient
    market_data_client: MarketDataClient
    registry: Registry

    raw_portfolio: Optional[RawPortfolio] = None
    portfolio: Optional[Portfolio] = None
    warnings: List[str] = field(default_factory=list)
    universe: Optional[RiskFactorUniverse] = None
    factor_data: Optional[FactorDataSet] = None
    elementary: Optional[ElementaryUniverse] = None
    base_prices: Optional[BasePriceSet] = None
    pnl: Optional[PnLResult] = None
    clusters: Optional[ClusterSet] = None

    timings_ms: dict = field(default_factory=dict)

    @property
    def cob_date(self) -> str:
        return self.config.cob_date
```

### `pipeline/engine/__init__.py`

```python
"""Compute engines (the threaded PnL engine)."""
from src.rade_static_replication.pipeline.engine.pnl_engine import run_pnl_engine

__all__ = ["run_pnl_engine"]
```

### `pipeline/engine/pnl_engine.py`

```python
"""
PnL engine.

Computes each factor's elementary scenario-PnL matrix by routing to its pricer's
vectorised ``scenario_pnl``. Per-factor work is independent, so factors fan out across a
thread pool (the heavy numerics run in compiled kernels that release the GIL when Numba
is present; otherwise it degrades to serial-equivalent throughput).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.domain.contracts import (
    BasePriceSet,
    ElementaryUniverse,
    FactorDataSet,
    FactorPnL,
    PnLResult,
)
from src.rade_static_replication.domain.errors import PnLError

logger = logging.getLogger(__name__)


def _price_one_factor(factor_id, trades, factor_data, base_prices, registry) -> FactorPnL:
    """Price one factor's elementary trades across all scenarios."""
    plugin = registry.get(factor_data.spec.asset_class)
    pnl_matrix = plugin.pricer.scenario_pnl(trades, factor_data, base_prices)
    return FactorPnL(
        factor_id=factor_id,
        trade_ids=[trade.trade_id for trade in trades],
        scenario_ids=factor_data.scenarios.scenario_ids,
        pnl=pnl_matrix,
    )


def run_pnl_engine(
    elementary: ElementaryUniverse,
    factor_data: FactorDataSet,
    base: BasePriceSet,
    registry: Registry,
    max_workers: int = 1,
) -> PnLResult:
    """Compute the elementary scenario PnL for every primary factor."""
    # Only factors that actually produced elementary trades need pricing.
    factor_ids = [fid for fid in elementary.by_factor if elementary.by_factor[fid]]

    def _compute(factor_id: str) -> FactorPnL:
        try:
            return _price_one_factor(
                factor_id, elementary.by_factor[factor_id],
                factor_data.factors[factor_id], base.by_factor[factor_id], registry,
            )
        except Exception as exc:  # attach the offending factor for a clear failure
            raise PnLError(f"PnL failed for factor {factor_id}: {exc}") from exc

    results: Dict[str, FactorPnL] = {}
    if max_workers > 1 and len(factor_ids) > 1:
        # Factors are independent -> fan out across threads.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for factor_pnl in pool.map(_compute, factor_ids):
                results[factor_pnl.factor_id] = factor_pnl
    else:
        for factor_id in factor_ids:
            factor_pnl = _compute(factor_id)
            results[factor_pnl.factor_id] = factor_pnl

    logger.info("Computed PnL for %d factors", len(results))
    return PnLResult(by_factor=results)
```

### `pipeline/orchestrator.py`

```python
"""
Orchestrator — sequences the pipeline and owns the audit trail.

Each stage method reads contracts off the :class:`RunContext`, runs one stage, records
its output + timing, and returns it. :meth:`run` executes the full sequence and writes
the auditable artifact store; individual stages can also be driven by hand for
debugging or partial runs.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Optional

from src.rade_static_replication import __version__
from src.rade_static_replication.artifacts.store import RunArtifactStore
from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.assets.registry import default_registry
from src.rade_static_replication.clients.base import MarketDataClient, PortfolioClient
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import ArtifactManifest
from src.rade_static_replication.pipeline import stages
from src.rade_static_replication.pipeline.context import RunContext

logger = logging.getLogger(__name__)


class Orchestrator:
    """Drives the static-replication preprocessing pipeline end to end."""

    def __init__(
        self,
        config: OrchestratorConfig,
        portfolio_client: PortfolioClient,
        market_data_client: MarketDataClient,
        registry: Optional[Registry] = None,
    ) -> None:
        self.ctx = RunContext(
            config=config,
            portfolio_client=portfolio_client,
            market_data_client=market_data_client,
            registry=registry or default_registry(),
        )
        self.store = RunArtifactStore(config, library_version=__version__)

    @contextmanager
    def _timed(self, name: str):
        start = time.perf_counter()
        logger.info("stage start: %s", name)
        yield
        self.ctx.timings_ms[name] = (time.perf_counter() - start) * 1000.0

    # ---- individual stages ----

    def load(self) -> None:
        with self._timed("load"):
            self.ctx.raw_portfolio = stages.load_portfolio(self.ctx.portfolio_client, self.ctx.cob_date)

    def normalise(self) -> None:
        with self._timed("normalise"):
            self.ctx.portfolio, self.ctx.warnings = stages.normalise_portfolio(self.ctx.raw_portfolio)

    def resolve(self) -> None:
        with self._timed("resolve"):
            self.ctx.universe = stages.resolve_universe(self.ctx.portfolio, self.ctx.config)

    def build(self) -> None:
        with self._timed("build_market"):
            self.ctx.factor_data = stages.build_factor_data(
                self.ctx.universe, self.ctx.registry, self.ctx.market_data_client, self.ctx.cob_date,
            )

    def generate(self) -> None:
        with self._timed("generate"):
            self.ctx.elementary = stages.generate_elementary(
                self.ctx.universe, self.ctx.factor_data, self.ctx.registry, self.ctx.config,
            )

    def price(self) -> None:
        with self._timed("price"):
            self.ctx.base_prices = stages.price_base(
                self.ctx.elementary, self.ctx.factor_data, self.ctx.registry,
            )

    def pnl(self) -> None:
        with self._timed("pnl"):
            self.ctx.pnl = stages.compute_pnl(
                self.ctx.elementary, self.ctx.factor_data, self.ctx.base_prices,
                self.ctx.registry, self.ctx.config.engine,
            )

    def cluster(self) -> None:
        with self._timed("cluster"):
            self.ctx.clusters = stages.resolve_clusters(
                self.ctx.portfolio, self.ctx.universe, self.ctx.config,
            )

    def write(self) -> ArtifactManifest:
        with self._timed("write"):
            entries = self.store.write_clusters(
                self.ctx.clusters, self.ctx.portfolio, self.ctx.pnl, self.ctx.elementary,
            )
        return ArtifactManifest(entries=entries, root=self.store.layout.run_dir)

    # ---- full run ----

    def run(self) -> RunContext:
        """Execute the full pipeline and write the artifact store."""
        self.store.begin()
        try:
            self.load()
            self.normalise()
            self.resolve()
            self.store.record_portfolio(self.ctx.portfolio, self.ctx.universe)
            self.build()
            self.generate()
            self.price()
            self.pnl()
            self.cluster()
            self.write()
            self.store.finalise(counts=self._counts(), timings_ms=self.ctx.timings_ms)
        except Exception as exc:
            self.store.fail(str(exc))
            raise
        return self.ctx

    def _counts(self) -> dict:
        return {
            "trades": len(self.ctx.portfolio.trade_ids) if self.ctx.portfolio else 0,
            "risk_factors": len(self.ctx.universe.specs) if self.ctx.universe else 0,
            "elementary_trades": self.ctx.elementary.n_trades if self.ctx.elementary else 0,
            "clusters": len(self.ctx.clusters.clusters) if self.ctx.clusters else 0,
        }
```

### `pipeline/stages/__init__.py`

```python
"""Pipeline stages — one thin, pure function per stage."""
from src.rade_static_replication.pipeline.stages.build_market import build_factor_data
from src.rade_static_replication.pipeline.stages.cluster import resolve_clusters
from src.rade_static_replication.pipeline.stages.generate import generate_elementary
from src.rade_static_replication.pipeline.stages.load import load_portfolio
from src.rade_static_replication.pipeline.stages.normalise import normalise_portfolio
from src.rade_static_replication.pipeline.stages.pnl import compute_pnl
from src.rade_static_replication.pipeline.stages.price import price_base
from src.rade_static_replication.pipeline.stages.resolve import resolve_universe

__all__ = [
    "load_portfolio", "normalise_portfolio", "resolve_universe", "build_factor_data",
    "generate_elementary", "price_base", "compute_pnl", "resolve_clusters",
]
```

### `pipeline/stages/build_market.py`

```python
"""
Stage 4 — build risk-factor market data.

Builds each factor in dependency-respecting order (so an FX factor receives its IR
dependency curves), routing to the registered asset-class builder.
"""
from __future__ import annotations

import logging
from typing import Dict

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.clients.base import MarketDataClient
from src.rade_static_replication.domain.contracts import (
    FactorDataSet,
    RiskFactorData,
    RiskFactorUniverse,
)

logger = logging.getLogger(__name__)


def build_factor_data(
    universe: RiskFactorUniverse, registry: Registry, client: MarketDataClient, cob_date: str,
) -> FactorDataSet:
    """Build every risk factor's snapshot + scenarios in dependency order."""
    built: Dict[str, RiskFactorData] = {}
    for fid in universe.build_order():
        spec = universe.specs[fid]
        plugin = registry.get(spec.asset_class)
        deps = {dep: built[dep] for dep in spec.dependencies if dep in built}
        built[fid] = plugin.builder.build(spec, deps, client, cob_date)
        logger.debug("built factor %s (%s)", fid, spec.asset_class)
    logger.info("Built %d risk-factor market objects", len(built))
    return FactorDataSet(factors=built)
```

### `pipeline/stages/cluster.py`

```python
"""
Stage 8 — resolve clusters.

Slices the portfolio into single-asset-class clusters keyed by the configured
attribute columns (``AssetClass`` is always the first key, guaranteeing no
cross-asset clusters). Each cluster carries its target trade ids and the union of
their primary risk factors.
"""
from __future__ import annotations

import logging
import re
from typing import List

from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import (
    ClusterSet,
    ClusterSpec,
    Portfolio,
    RiskFactorUniverse,
)

logger = logging.getLogger(__name__)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-")


def resolve_clusters(
    portfolio: Portfolio, universe: RiskFactorUniverse, config: OrchestratorConfig,
) -> ClusterSet:
    """Group trades into single-asset-class clusters by the configured keys."""
    keys = config.clustering.keys
    if "AssetClass" not in keys:
        keys = ["AssetClass"] + list(keys)

    attr = portfolio.attributes
    missing = [k for k in keys if k not in attr.columns]
    if missing:
        logger.warning("clustering keys not in attributes, ignored: %s", missing)
        keys = [k for k in keys if k in attr.columns]

    clusters: List[ClusterSpec] = []
    for key_tuple, group in attr.groupby(keys, sort=True):
        values = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
        key_map = {k: str(v) for k, v in zip(keys, values)}
        trade_ids = [str(t) for t in group.index]

        factor_ids: List[str] = []
        for t in trade_ids:
            for fid in universe.factors_by_trade.get(t, []):
                if fid not in factor_ids:
                    factor_ids.append(fid)

        cluster_id = "__".join(_slug(v) for v in values)
        clusters.append(ClusterSpec(
            cluster_id=cluster_id, key=key_map, asset_class=key_map.get("AssetClass", ""),
            risk_factor_ids=factor_ids, target_trade_ids=trade_ids,
        ))

    logger.info("Resolved %d clusters", len(clusters))
    return ClusterSet(clusters=clusters)
```

### `pipeline/stages/generate.py`

```python
"""
Stage 5 — generate the elementary-trade universe.

Builds a replicating-basis grid for each *primary* factor (dependency-only factors are
market data, not replication targets), using that asset class's generator.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import (
    ElementaryUniverse,
    FactorDataSet,
    RiskFactorUniverse,
)
from src.rade_static_replication.domain.instruments import ElementaryTrade

logger = logging.getLogger(__name__)


def generate_elementary(
    universe: RiskFactorUniverse,
    factor_data: FactorDataSet,
    registry: Registry,
    config: OrchestratorConfig,
) -> ElementaryUniverse:
    """Emit the elementary universe for each primary risk factor."""
    by_factor: Dict[str, List[ElementaryTrade]] = {}
    for fid in universe.primary_ids:
        rf = factor_data.factors[fid]
        plugin = registry.get(rf.spec.asset_class)
        grid = config.elementary.grids.get(rf.spec.asset_class, {})
        by_factor[fid] = plugin.generator.generate(rf, grid)

    universe_out = ElementaryUniverse(by_factor=by_factor)
    logger.info(
        "Generated %d elementary trades across %d factors", universe_out.n_trades, len(by_factor)
    )
    return universe_out
```

### `pipeline/stages/load.py`

```python
"""Stage 1 — load the raw portfolio via the portfolio client."""
from __future__ import annotations

from src.rade_static_replication.clients.base import PortfolioClient
from src.rade_static_replication.domain.contracts import RawPortfolio


def load_portfolio(client: PortfolioClient, cob_date: str) -> RawPortfolio:
    """Fetch the untouched portfolio payload from the client."""
    return client.load(cob_date)
```

### `pipeline/stages/normalise.py`

```python
"""Stage 2 — normalise + validate the portfolio."""
from __future__ import annotations

from typing import List, Tuple

from src.rade_static_replication.domain.contracts import Portfolio, RawPortfolio
from src.rade_static_replication.portfolio.normalise import normalise as _normalise
from src.rade_static_replication.portfolio.validate import validate as _validate


def normalise_portfolio(raw: RawPortfolio) -> Tuple[Portfolio, List[str]]:
    """Collapse the exploded export to one row per trade and validate it."""
    portfolio = _normalise(raw)
    warnings = _validate(portfolio)
    return portfolio, warnings
```

### `pipeline/stages/pnl.py`

```python
"""Stage 7 — scenario PnL (delegates to the engine)."""
from __future__ import annotations

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.config.schema import EngineConfig
from src.rade_static_replication.domain.contracts import (
    BasePriceSet,
    ElementaryUniverse,
    FactorDataSet,
    PnLResult,
)
from src.rade_static_replication.pipeline.engine.pnl_engine import run_pnl_engine


def compute_pnl(
    elementary: ElementaryUniverse,
    factor_data: FactorDataSet,
    base: BasePriceSet,
    registry: Registry,
    engine: EngineConfig,
) -> PnLResult:
    """Compute per-factor elementary scenario PnL matrices."""
    return run_pnl_engine(elementary, factor_data, base, registry, max_workers=engine.max_workers)
```

### `pipeline/stages/price.py`

```python
"""Stage 6 — price the elementary universe at COB (base PVs)."""
from __future__ import annotations

from typing import Dict

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.domain.contracts import (
    BasePriceSet,
    ElementaryUniverse,
    FactorDataSet,
)


def price_base(
    elementary: ElementaryUniverse, factor_data: FactorDataSet, registry: Registry,
) -> BasePriceSet:
    """Compute COB present values for every elementary trade."""
    by_factor: Dict[str, Dict[str, float]] = {}
    for fid, trades in elementary.by_factor.items():
        rf = factor_data.factors[fid]
        plugin = registry.get(rf.spec.asset_class)
        by_factor[fid] = plugin.pricer.base_prices(trades, rf)
    return BasePriceSet(by_factor=by_factor)
```

### `pipeline/stages/resolve.py`

```python
"""Stage 3 — resolve the risk-factor universe."""
from __future__ import annotations

from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import Portfolio, RiskFactorUniverse
from src.rade_static_replication.portfolio.resolution import resolve as _resolve


def resolve_universe(portfolio: Portfolio, config: OrchestratorConfig) -> RiskFactorUniverse:
    """Resolve every primary + dependency risk factor for the portfolio."""
    return _resolve(portfolio, config)
```

### `artifacts/__init__.py`

```python
"""Auditable artifact store: run layout, manifest, writers."""
from src.rade_static_replication.artifacts.layout import RunLayout
from src.rade_static_replication.artifacts.manifest import RunManifest
from src.rade_static_replication.artifacts.store import RunArtifactStore

__all__ = ["RunLayout", "RunManifest", "RunArtifactStore"]
```

### `artifacts/layout.py`

```python
"""
Artifact directory contract.

One :class:`RunLayout` owns every path a run writes, so the on-disk structure is
defined in exactly one place and is trivially auditable:

    <root>/<run_id>/
        run_manifest.json     config.snapshot.yaml     logs/run.log
        portfolio/attributes.parquet     portfolio/universe.json
        clusters/<cluster_id>/{elementary_pnl.parquet, elementary_attributes.pkl,
                               target_pnl.parquet, target_attributes.pkl}
        jobs.pkl
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.rade_static_replication.domain.contracts import ArtifactPaths


@dataclass(frozen=True)
class RunLayout:
    """Resolves every path for a single run."""
    root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def manifest_json(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def config_snapshot(self) -> Path:
        return self.run_dir / "config.snapshot.yaml"

    @property
    def log_file(self) -> Path:
        return self.run_dir / "logs" / "run.log"

    @property
    def portfolio_dir(self) -> Path:
        return self.run_dir / "portfolio"

    @property
    def attributes_parquet(self) -> Path:
        return self.portfolio_dir / "attributes.parquet"

    @property
    def universe_json(self) -> Path:
        return self.portfolio_dir / "universe.json"

    @property
    def clusters_dir(self) -> Path:
        return self.run_dir / "clusters"

    @property
    def jobs_pkl(self) -> Path:
        return self.run_dir / "jobs.pkl"

    def cluster_dir(self, cluster_id: str) -> Path:
        return self.clusters_dir / cluster_id

    def cluster_paths(self, cluster_id: str) -> ArtifactPaths:
        d = self.cluster_dir(cluster_id)
        return ArtifactPaths(
            elementary_pnl=d / "elementary_pnl.parquet",
            elementary_attributes=d / "elementary_attributes.pkl",
            target_pnl=d / "target_pnl.parquet",
            target_attributes=d / "target_attributes.pkl",
        )

    def ensure_base_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)
        self.clusters_dir.mkdir(parents=True, exist_ok=True)
```

### `artifacts/manifest.py`

```python
"""
Run manifest — the audit header for a run.

Captures everything needed to reproduce/trace a run: library version, git sha,
timestamps, cob date, counts, and per-cluster file pointers. Written as JSON next to
the artifacts.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def git_sha() -> str:
    """Best-effort current commit sha (``"unknown"`` if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class RunManifest:
    """Audit metadata for one run."""
    run_id: str
    cob_date: str
    library_version: str
    git_sha: str
    created_utc: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    status: str = "running"
    counts: Dict[str, int] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    clusters: List[Dict[str, Any]] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
```

### `artifacts/store.py`

```python
"""
Run artifact store — the auditable run ledger.

Owns the lifecycle of one run's output directory: create it, snapshot the config,
persist the normalised portfolio + resolved universe (audit copies), write the
cluster files, and finalise the manifest. One object, one run, full traceability.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from src.rade_static_replication.artifacts.layout import RunLayout
from src.rade_static_replication.artifacts.manifest import RunManifest, git_sha
from src.rade_static_replication.artifacts.writers import write_artifacts
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import (
    ClusterSet,
    ElementaryUniverse,
    PnLResult,
    Portfolio,
    RiskFactorUniverse,
)

logger = logging.getLogger(__name__)


def _config_to_dict(config: OrchestratorConfig) -> dict:
    return {
        "cob_date": config.cob_date,
        "factors": {
            ac: {
                "source": r.source, "key_field": r.key_field,
                "factor_fields": r.factor_fields, "dependency_fields": r.dependency_fields,
                "asset_class_of": r.asset_class_of, "mapping_path": r.mapping_path,
            }
            for ac, r in config.factors.rules.items()
        },
        "elementary": config.elementary.grids,
        "engine": {"max_workers": config.engine.max_workers, "use_numba": config.engine.use_numba},
        "clustering": {"keys": config.clustering.keys},
        "output": {"root": str(config.output.root), "run_id": config.output.run_id},
    }


class RunArtifactStore:
    """Lifecycle manager for one run's artifacts."""

    def __init__(self, config: OrchestratorConfig, library_version: str) -> None:
        self.config = config
        self.layout = RunLayout(root=Path(config.output.root), run_id=config.output.run_id)
        self.manifest = RunManifest(
            run_id=config.output.run_id, cob_date=config.cob_date,
            library_version=library_version, git_sha=git_sha(),
        )

    def begin(self) -> "RunArtifactStore":
        import yaml

        self.layout.ensure_base_dirs()
        self.layout.config_snapshot.write_text(yaml.safe_dump(_config_to_dict(self.config), sort_keys=False))
        self.manifest.write(self.layout.manifest_json)
        logger.info("artifact run dir: %s", self.layout.run_dir)
        return self

    def record_portfolio(self, portfolio: Portfolio, universe: RiskFactorUniverse) -> None:
        portfolio.attributes.to_parquet(self.layout.attributes_parquet)
        payload = {
            "factors": {
                fid: {
                    "asset_class": s.asset_class, "is_primary": s.is_primary,
                    "dependencies": list(s.dependencies), "meta": s.meta,
                }
                for fid, s in universe.specs.items()
            },
            "factors_by_trade": universe.factors_by_trade,
        }
        self.layout.universe_json.write_text(json.dumps(payload, indent=2, default=str))

    def write_clusters(
        self, clusters: ClusterSet, portfolio: Portfolio, pnl: PnLResult, universe: ElementaryUniverse,
    ) -> List[Dict]:
        entries = write_artifacts(self.layout, clusters, portfolio, pnl, universe)
        self.manifest.clusters = entries
        return entries

    def finalise(self, counts: Dict[str, int], timings_ms: Dict[str, float]) -> None:
        self.manifest.status = "completed"
        self.manifest.counts = counts
        self.manifest.timings_ms = timings_ms
        self.manifest.write(self.layout.manifest_json)
        logger.info("run %s finalised", self.manifest.run_id)

    def fail(self, reason: str) -> None:
        self.manifest.status = f"failed: {reason}"
        self.manifest.write(self.layout.manifest_json)
```

### `artifacts/writers.py`

```python
"""
Artifact writers.

Materialise the per-cluster files in the exact layout ``rade_ml`` consumes:

* ``elementary_pnl.parquet`` / ``target_pnl.parquet`` — ``[scenarios × trade-ids]``;
* ``elementary_attributes.pkl`` / ``target_attributes.pkl`` — column-oriented dicts;
* ``jobs.pkl`` — the run manifest of cluster paths.

Pure persistence: it receives resolved contracts and writes them. The one piece of logic
is aligning the (market) elementary scenario axis with the (portfolio) target scenario
axis so the two PnL frames share rows.
"""
from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List

import pandas as pd

from src.rade_static_replication.artifacts.layout import RunLayout
from src.rade_static_replication.domain.contracts import (
    ClusterSet,
    ElementaryUniverse,
    PnLResult,
    Portfolio,
)

logger = logging.getLogger(__name__)


def _elementary_pnl_frame(cluster_factor_ids, pnl: PnLResult) -> pd.DataFrame:
    """Stack a cluster's elementary PnL into ``[scenarios × elementary-trade-ids]``."""
    per_factor_frames: List[pd.DataFrame] = []
    for factor_id in cluster_factor_ids:
        factor_pnl = pnl.by_factor.get(factor_id)
        if factor_pnl is None:
            continue
        # Stored as (trades, scenarios); transpose to scenarios-on-rows for rade_ml.
        per_factor_frames.append(pd.DataFrame(
            factor_pnl.pnl.T,
            index=[str(s) for s in factor_pnl.scenario_ids],
            columns=factor_pnl.trade_ids,
        ))
    if not per_factor_frames:
        return pd.DataFrame()
    return pd.concat(per_factor_frames, axis=1)


def _elementary_attributes(cluster_factor_ids, universe: ElementaryUniverse) -> Dict[str, list]:
    """Column-oriented attribute dict describing each elementary trade in the cluster."""
    rows: List[dict] = []
    for factor_id in cluster_factor_ids:
        for trade in universe.by_factor.get(factor_id, []):
            rows.append({
                "trade_id": trade.trade_id, "factor_id": trade.factor_id,
                "asset_class": trade.asset_class, "payoff_type": trade.payoff_type,
                **{f"param_{name}": value for name, value in trade.parameters.items()},
            })
    attributes = pd.DataFrame(rows)
    return {column: attributes[column].tolist() for column in attributes.columns}


def write_artifacts(
    layout: RunLayout,
    clusters: ClusterSet,
    portfolio: Portfolio,
    pnl: PnLResult,
    universe: ElementaryUniverse,
) -> List[Dict[str, Any]]:
    """Write every cluster's four files + return the ``jobs.pkl`` manifest entries."""
    manifest_entries: List[Dict[str, Any]] = []

    # Targets are stored [trade × scenario]; transpose once to [scenario × trade].
    target_pnl_all = portfolio.target_pnl.T
    target_pnl_all.index = [str(s) for s in target_pnl_all.index]

    for cluster in clusters.clusters:
        paths = layout.cluster_paths(cluster.cluster_id)
        paths.elementary_pnl.parent.mkdir(parents=True, exist_ok=True)

        elementary_pnl = _elementary_pnl_frame(cluster.risk_factor_ids, pnl)
        target_pnl = target_pnl_all.reindex(
            columns=[t for t in cluster.target_trade_ids if t in target_pnl_all.columns]
        )

        # Align scenario axes: market scenarios ∩ portfolio scenarios, in market order.
        if not elementary_pnl.empty:
            shared_scenarios = [s for s in elementary_pnl.index if s in set(target_pnl.index)]
            if shared_scenarios:
                elementary_pnl = elementary_pnl.reindex(index=shared_scenarios)
                target_pnl = target_pnl.reindex(index=shared_scenarios)

        elementary_pnl.to_parquet(paths.elementary_pnl)
        target_pnl.to_parquet(paths.target_pnl)

        with open(paths.elementary_attributes, "wb") as file_handle:
            pickle.dump(_elementary_attributes(cluster.risk_factor_ids, universe), file_handle)

        target_attributes = portfolio.attributes.loc[
            [t for t in cluster.target_trade_ids if t in portfolio.attributes.index]
        ]
        with open(paths.target_attributes, "wb") as file_handle:
            pickle.dump(
                {"trade_id": target_attributes.index.tolist(),
                 **{column: target_attributes[column].tolist() for column in target_attributes.columns}},
                file_handle,
            )

        manifest_entries.append({
            "cluster_id": cluster.cluster_id, "key": cluster.key, "asset_class": cluster.asset_class,
            "n_elementary": int(elementary_pnl.shape[1]), "n_targets": int(target_pnl.shape[1]),
            "n_scenarios": int(elementary_pnl.shape[0]),
            "elementary_pnl": str(paths.elementary_pnl),
            "elementary_attributes": str(paths.elementary_attributes),
            "target_pnl": str(paths.target_pnl),
            "target_attributes": str(paths.target_attributes),
        })
        logger.info(
            "cluster %s: %d elementary x %d targets x %d scenarios",
            cluster.cluster_id, elementary_pnl.shape[1], target_pnl.shape[1], elementary_pnl.shape[0],
        )

    with open(layout.jobs_pkl, "wb") as file_handle:
        pickle.dump(manifest_entries, file_handle)
    return manifest_entries
```

### `clients/__init__.py`

```python
"""
Client layer (infrastructure / ports & adapters).

The pipeline depends only on the two *ports* (:class:`PortfolioClient`,
:class:`MarketDataClient`) and the payload dataclasses. Concrete adapters
(``mock``, ``file``, ``api``) are imported explicitly by the caller, keeping this
package import light.
"""
from src.rade_static_replication.clients.base import MarketDataClient, PortfolioClient
from src.rade_static_replication.clients.payloads import (
    CubePayload,
    CurvePayload,
    FXShockPayload,
    RatesShockPayload,
    SurfacePayload,
)

__all__ = [
    "PortfolioClient", "MarketDataClient",
    "CurvePayload", "SurfacePayload", "CubePayload", "FXShockPayload", "RatesShockPayload",
]
```

### `clients/base.py`

```python
"""
Client ports — the abstract interfaces the pipeline depends on.

The pipeline never touches Sage/STAR/CSV directly; it depends on these two
protocols. Any object with the right methods works (mock, file, or your API
adapters). This is the dependency-inversion seam: swap data source without touching
business logic.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from src.rade_static_replication.clients.payloads import (
    CubePayload,
    CurvePayload,
    FXShockPayload,
    RatesShockPayload,
    SurfacePayload,
)
from src.rade_static_replication.domain.contracts import RawPortfolio


@runtime_checkable
class PortfolioClient(Protocol):
    """Source of the target portfolio (trade attributes + scenario PnL)."""

    def load(self, cob_date: str) -> RawPortfolio:
        ...


@runtime_checkable
class MarketDataClient(Protocol):
    """Source of COB market data and scenario shocks, per risk factor."""

    def fx_spot(self, pair: str, cob_date: str) -> float: ...

    def discount_curve(self, currency: str, cob_date: str) -> CurvePayload: ...

    def fx_vol_surface(self, pair: str, cob_date: str) -> SurfacePayload: ...

    def fx_shocks(self, pair: str, cob_date: str) -> FXShockPayload: ...

    def rates_vol_cube(self, factor_id: str, cob_date: str) -> Optional[CubePayload]: ...

    def rates_shocks(self, factor_id: str, cob_date: str) -> RatesShockPayload: ...
```

### `clients/file/__init__.py`

```python
"""File-based clients (read exports from disk)."""
from src.rade_static_replication.clients.file.marketdata import FileMarketDataClient
from src.rade_static_replication.clients.file.portfolio import FilePortfolioClient

__all__ = ["FilePortfolioClient", "FileMarketDataClient"]
```

### `clients/file/portfolio.py`

```python
"""
File-based portfolio client.

Reads the raw exploded attributes + scenario PnL from CSV/parquet on disk (e.g. an
export dumped from Sage). Ready to use as-is.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.rade_static_replication.domain.contracts import RawPortfolio
from src.rade_static_replication.domain.errors import ClientError


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ClientError(f"portfolio file not found: {path}")
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


class FilePortfolioClient:
    """Load the portfolio from local files.

    Parameters
    ----------
    attributes_path, pnl_path : str | Path
        Raw exploded attributes and scenario-PnL files.
    pnl_index_col : str
        Trade-id column in the PnL file (default ``"trade_id"``).
    """

    def __init__(self, attributes_path, pnl_path, pnl_index_col: str = "trade_id") -> None:
        self.attributes_path = Path(attributes_path)
        self.pnl_path = Path(pnl_path)
        self.pnl_index_col = pnl_index_col

    def load(self, cob_date: str) -> RawPortfolio:
        attributes = _read(self.attributes_path)
        pnl = _read(self.pnl_path)
        if self.pnl_index_col in pnl.columns:
            pnl = pnl.set_index(self.pnl_index_col)
        return RawPortfolio(attributes=attributes, target_pnl=pnl, cob_date=cob_date, source=str(self.attributes_path))
```

### `clients/mock/__init__.py`

```python
"""Mock clients for offline development and tests."""
from src.rade_static_replication.clients.mock.marketdata import MockMarketDataClient
from src.rade_static_replication.clients.mock.portfolio import MockPortfolioClient

__all__ = ["MockPortfolioClient", "MockMarketDataClient"]
```

### `clients/mock/marketdata.py`

```python
"""
Mock market-data client.

Deterministic, internally-consistent FX/Rates market data + shocks for offline
development and tests. Seeded per lookup key so runs are reproducible. Replace with the
STAR adapter in production — the **array shapes returned here are the contract** your
adapter must satisfy (diff your real payloads against these).
"""
from __future__ import annotations

import numpy as np

from src.rade_static_replication.clients.payloads import (
    CubePayload,
    CurvePayload,
    FXShockPayload,
    RatesShockPayload,
    SurfacePayload,
)

# Shared grids (the axis labels every payload is built on).
CURVE_TENORS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
VOL_MONEYNESS = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
VOL_EXPIRIES = np.array([0.25, 0.5, 1.0, 2.0])
CUBE_EXPIRIES = np.array([1.0, 2.0, 5.0])
CUBE_TENORS = np.array([2.0, 5.0, 10.0])
CUBE_STRIKES = np.array([0.0, 0.02, 0.04, 0.06, 0.08])
N_SCENARIOS = 60

# Plausible COB levels keyed by parser output (foreign+domestic) / currency.
_BASE_SPOT = {"EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 150.0, "AUDUSD": 0.66}
_BASE_RATE = {"USD": 0.045, "EUR": 0.032, "GBP": 0.047, "JPY": 0.004}


def _seeded_rng(*key) -> np.random.Generator:
    """A reproducible RNG derived from a lookup key (so a pair/ccy always yields the same data)."""
    return np.random.default_rng(abs(hash(key)) % (2**32))


class MockMarketDataClient:
    """Generates seeded, consistent market data + shocks."""

    def __init__(self, n_scenarios: int = N_SCENARIOS) -> None:
        self.n_scenarios = n_scenarios
        self.scenario_ids = np.array([f"scen_{i}" for i in range(self.n_scenarios)])

    # ---- FX ----

    def fx_spot(self, pair: str, cob_date: str) -> float:
        return _BASE_SPOT.get(pair.upper(), 1.0)

    def fx_vol_surface(self, pair: str, cob_date: str) -> SurfacePayload:
        rng = _seeded_rng("fxvol", pair)
        atm_vol = 0.08 + 0.04 * rng.random()
        smile = 0.02 * (VOL_MONEYNESS - 1.0) ** 2 / 0.01      # convex in moneyness
        term_factor = 1.0 + 0.1 * np.log1p(VOL_EXPIRIES)      # mild upward term structure
        vols = atm_vol * np.outer(term_factor, np.ones_like(VOL_MONEYNESS)) + smile[None, :]
        return SurfacePayload(VOL_EXPIRIES, VOL_MONEYNESS, vols, "moneyness", "lognormal")

    def fx_shocks(self, pair: str, cob_date: str) -> FXShockPayload:
        rng = _seeded_rng("fxshock", pair)
        n = self.n_scenarios
        spot = self.fx_spot(pair, cob_date) * np.exp(rng.normal(0, 0.01, n))          # lognormal spot moves
        base_surface = self.fx_vol_surface(pair, cob_date).vols
        vol = base_surface[None, :, :] * np.exp(rng.normal(0, 0.05, (n, *base_surface.shape)))

        foreign_ccy = pair[:3].upper()
        domestic_curve = self.discount_curve("USD", cob_date)
        foreign_curve = self.discount_curve(foreign_ccy, cob_date)
        domestic_rate = domestic_curve.zero_rates[None, :] + rng.normal(0, 5e-4, (n, CURVE_TENORS.size))
        foreign_rate = foreign_curve.zero_rates[None, :] + rng.normal(0, 5e-4, (n, CURVE_TENORS.size))
        return FXShockPayload(
            scenario_ids=self.scenario_ids, spot=spot, vol=vol,
            vol_expiries=VOL_EXPIRIES, vol_strikes=VOL_MONEYNESS,
            domestic_rate=domestic_rate, domestic_tenors=CURVE_TENORS,
            foreign_rate=foreign_rate, foreign_tenors=CURVE_TENORS,
        )

    # ---- Rates / shared ----

    def discount_curve(self, currency: str, cob_date: str) -> CurvePayload:
        rng = _seeded_rng("curve", currency)
        level = _BASE_RATE.get(currency.upper(), 0.03)
        slope = 0.004 * rng.random()
        zero_rates = level + slope * np.log1p(CURVE_TENORS)   # gently upward-sloping curve
        return CurvePayload(CURVE_TENORS, zero_rates)

    def rates_vol_cube(self, factor_id: str, cob_date: str) -> CubePayload:
        rng = _seeded_rng("cube", factor_id)
        atm_normal_vol = 0.006 + 0.003 * rng.random()
        skew = 0.0005 * (CUBE_STRIKES - 0.04) / 0.02
        vols = atm_normal_vol + skew[None, None, :] + np.zeros(
            (CUBE_EXPIRIES.size, CUBE_TENORS.size, CUBE_STRIKES.size)
        )
        return CubePayload(CUBE_EXPIRIES, CUBE_TENORS, CUBE_STRIKES, np.abs(vols), "normal")

    def rates_shocks(self, factor_id: str, cob_date: str) -> RatesShockPayload:
        rng = _seeded_rng("rshock", factor_id)
        currency = factor_id.split(".")[-1]
        base_curve = self.discount_curve(currency, cob_date).zero_rates
        n = self.n_scenarios
        curve = base_curve[None, :] + rng.normal(0, 8e-4, (n, base_curve.size))       # additive rate shocks
        base_cube = self.rates_vol_cube(factor_id, cob_date).vols
        vol = np.abs(base_cube[None, ...] + rng.normal(0, 5e-4, (n, *base_cube.shape)))
        return RatesShockPayload(
            scenario_ids=self.scenario_ids, curve=curve, curve_tenors=CURVE_TENORS,
            vol=vol, vol_expiries=CUBE_EXPIRIES, vol_swap_tenors=CUBE_TENORS, vol_strikes=CUBE_STRIKES,
        )
```

### `clients/mock/portfolio.py`

```python
"""
Mock portfolio client.

Generates a raw, *exploded* FX + Rates portfolio in the exact shape of the external
export (one row per trade × risk-type, notional row tagged ``AssetClass="ALL"``) plus
a scenario PnL frame. Replace with the Sage adapter in production.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.rade_static_replication.domain.contracts import RawPortfolio

_FX_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD"]
_RATES_FACTORS = ["IR_CURVE_SWAP.USD", "IR_CURVE_SWAP.EUR", "IR_CURVE_SWAP.GBP"]


class MockPortfolioClient:
    """Produce a representative cross-asset desk portfolio."""

    def __init__(self, n_fx: int = 6, n_rates: int = 4, n_scenarios: int = 60, seed: int = 7) -> None:
        self.n_fx = n_fx
        self.n_rates = n_rates
        self.n_scenarios = n_scenarios
        self.rng = np.random.default_rng(seed)

    def load(self, cob_date: str) -> RawPortfolio:
        rows: List[dict] = []
        trade_ids: List[str] = []
        asset_of: dict[str, str] = {}
        tid = 0

        for i in range(self.n_fx):
            pair = _FX_PAIRS[i % len(_FX_PAIRS)]
            t = f"FX{tid:04d}"
            tid += 1
            trade_ids.append(t)
            asset_of[t] = "FX"
            bs = "Buy" if self.rng.random() > 0.5 else "Sell"
            mat = ["20270615", "20280615", "20290615"][i % 3]
            for rt, val, ac in [
                ("FXPV", self.rng.normal(0, 1e5), "FX"),
                ("FXPOS", self.rng.normal(0, 5e5), "FX"),
                ("Notional", abs(self.rng.normal(1e7, 2e6)), "ALL"),
            ]:
                rows.append({
                    "PTSDealNumber": t, "AssetClass": ac, "RiskType": rt,
                    "Sum_RiskValuesUSD_Net": val, "BuySellInd": bs,
                    "MaturityDate": mat, "DeskName": "G10_FX",
                    "Product": "VanillaOption", "ccy": pair, "RatesFactor": "",
                })

        for i in range(self.n_rates):
            factor = _RATES_FACTORS[i % len(_RATES_FACTORS)]
            t = f"IR{tid:04d}"
            tid += 1
            trade_ids.append(t)
            asset_of[t] = "RATES"
            bs = "Buy" if self.rng.random() > 0.5 else "Sell"
            mat = ["20300615", "20340615"][i % 2]
            for rt, val, ac in [
                ("IRPV", self.rng.normal(0, 2e5), "RATES"),
                ("Notional", abs(self.rng.normal(5e7, 1e7)), "ALL"),
            ]:
                rows.append({
                    "PTSDealNumber": t, "AssetClass": ac, "RiskType": rt,
                    "Sum_RiskValuesUSD_Net": val, "BuySellInd": bs,
                    "MaturityDate": mat, "DeskName": f"Rates_{factor.split('.')[-1]}",
                    "Product": "Swaption", "ccy": "", "RatesFactor": factor,
                })

        attributes = pd.DataFrame(rows)

        scen_cols = [f"scen_{i}" for i in range(self.n_scenarios)]
        pnl = self.rng.normal(0, 5e4, (len(trade_ids), self.n_scenarios))
        target = pd.DataFrame(pnl, index=trade_ids, columns=scen_cols)
        target.index.name = "trade_id"
        target["AssetClass"] = [asset_of[t] for t in trade_ids]

        return RawPortfolio(attributes=attributes, target_pnl=target, cob_date=cob_date, source="mock")
```

### `clients/payloads.py`

```python
"""
Raw client payloads — the numeric wire format clients return.

These are deliberately *thin* (arrays + axis labels), with no behaviour: clients
produce payloads, the asset builders turn them into validated market objects. This
keeps client wiring trivial and the validation in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CurvePayload:
    """Zero-rate curve pillars."""
    tenors: np.ndarray
    zero_rates: np.ndarray


@dataclass(frozen=True)
class SurfacePayload:
    """FX vol surface grid."""
    expiries: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    strike_convention: str = "moneyness"
    vol_type: str = "lognormal"


@dataclass(frozen=True)
class CubePayload:
    """Swaption vol cube grid (normal vols)."""
    expiries: np.ndarray
    swap_tenors: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    vol_type: str = "normal"


@dataclass(frozen=True)
class FXShockPayload:
    """Resolved (absolute) FX scenario states aligned to scenario ids."""
    scenario_ids: np.ndarray
    spot: np.ndarray
    vol: np.ndarray
    vol_expiries: np.ndarray
    vol_strikes: np.ndarray
    domestic_rate: np.ndarray
    domestic_tenors: np.ndarray
    foreign_rate: np.ndarray
    foreign_tenors: np.ndarray


@dataclass(frozen=True)
class RatesShockPayload:
    """Resolved (absolute) rates scenario states."""
    scenario_ids: np.ndarray
    curve: np.ndarray
    curve_tenors: np.ndarray
    vol: Optional[np.ndarray] = None
    vol_expiries: Optional[np.ndarray] = None
    vol_swap_tenors: Optional[np.ndarray] = None
    vol_strikes: Optional[np.ndarray] = None
```

### `configs/fx_mapping.csv`

```csv
ccy,fx_risk_factor_1,fx_risk_factor_2,ir_factor_1,ir_factor_2
EURUSD,FX.SPOT.USD.EUR,FX.SPOT.USD.USD,IR_CURVE_SWAP.EUR,IR_CURVE_SWAP.USD
GBPUSD,FX.SPOT.USD.GBP,FX.SPOT.USD.USD,IR_CURVE_SWAP.GBP,IR_CURVE_SWAP.USD
AUDUSD,FX.SPOT.USD.AUD,FX.SPOT.USD.USD,IR_CURVE_SWAP.AUD,IR_CURVE_SWAP.USD
USDJPY,FX.SPOT.USD.JPY,FX.SPOT.USD.USD,IR_CURVE_SWAP.JPY,IR_CURVE_SWAP.USD
```

### `configs/orchestrator.yaml`

```yaml
# Example OrchestratorConfig. Block order mirrors the pipeline stages.
cob_date: "20240102"

# Stage 3 — risk-factor resolution rules, per asset class.
factors:
  fx:
    source: mapping            # look the key up in a CSV
    key_field: ccy             # portfolio column holding the pair (e.g. EURUSD)
    mapping_path: fx_mapping.csv   # resolved relative to this file
    factor_fields: [fx_risk_factor_1]                 # -> primary risk factor(s)
    dependency_fields: [fx_risk_factor_2, ir_factor_1, ir_factor_2]
    asset_class_of:            # asset class produced by each field
      fx_risk_factor_1: fx
      fx_risk_factor_2: fx
      ir_factor_1: rates
      ir_factor_2: rates
  rates:
    source: attribute          # factor id already on the trade row
    key_field: ""
    factor_fields: [RatesFactor]

# Stage 5 — elementary grids, per asset class (passed verbatim to the generator).
elementary:
  fx:
    moneyness: [0.90, 0.95, 1.00, 1.05, 1.10]
    expiries: [0.25, 0.50, 1.00]
    instruments: [call, put, digital_call, forward]
  rates:
    expiries: [1.0, 2.0, 5.0]
    swap_tenors: [2.0, 5.0, 10.0]
    strike_offsets_bp: [-50, 0, 50]
    instruments: [payer, receiver]
    freq: 0.5

# Stage 7 — PnL engine.
engine:
  max_workers: 4
  use_numba: true

# Stage 8 — clustering keys (AssetClass is always applied first).
clustering:
  keys: [AssetClass, DeskName]

# Artifact store location: <root>/<run_id>/...
output:
  root: data/rade_static_replication
  run_id: mock_run
```

### `examples/__init__.py`

```python
"""Runnable examples."""
```

### `examples/run_pipeline_mock.py`

```python
"""
End-to-end example: run the full pipeline against the mock clients.

    python -m src.rade_static_replication.examples.run_pipeline_mock

Swap ``MockPortfolioClient`` / ``MockMarketDataClient`` for the Sage / STAR adapters
(or the file clients) and everything else stays the same.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.rade_static_replication import load_config, run
from src.rade_static_replication.clients.mock import MockMarketDataClient, MockPortfolioClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "orchestrator.yaml"


def main() -> None:
    config = load_config(CONFIG_PATH)
    ctx = run(config, MockPortfolioClient(), MockMarketDataClient())

    print("\n=== run summary ===")
    print(f"trades             : {len(ctx.portfolio.trade_ids)}")
    print(f"risk factors       : {len(ctx.universe.specs)} "
          f"({len(ctx.universe.primary_ids)} primary)")
    print(f"elementary trades  : {ctx.elementary.n_trades}")
    print(f"clusters           : {len(ctx.clusters.clusters)}")
    for c in ctx.clusters.clusters:
        print(f"  - {c.cluster_id}: {len(c.target_trade_ids)} targets, "
              f"{len(c.risk_factor_ids)} factors")
    print(f"artifacts          : {ctx.config.output.root}/{ctx.config.output.run_id}")
    print(f"timings (ms)       : { {k: round(v, 1) for k, v in ctx.timings_ms.items()} }")


if __name__ == "__main__":
    main()
```

### `tests/__init__.py`

```python

```

### `tests/integration/__init__.py`

```python

```

### `tests/integration/test_pipeline.py`

```python
"""Integration test: full pipeline against mock clients."""
import pickle

from src.rade_static_replication import run
from src.rade_static_replication.clients.mock import MockMarketDataClient, MockPortfolioClient
from src.rade_static_replication.config.schema import (
    AssetFactorRule,
    ClusteringConfig,
    ElementaryConfig,
    FactorResolutionConfig,
    OrchestratorConfig,
    OutputConfig,
)


def _config(tmp_path, mapping_csv) -> OrchestratorConfig:
    return OrchestratorConfig(
        cob_date="20240102",
        factors=FactorResolutionConfig(rules={
            "fx": AssetFactorRule(
                asset_class="fx", source="mapping", key_field="ccy",
                factor_fields=["fx_risk_factor_1"],
                dependency_fields=["fx_risk_factor_2", "ir_factor_1", "ir_factor_2"],
                asset_class_of={
                    "fx_risk_factor_1": "fx", "fx_risk_factor_2": "fx",
                    "ir_factor_1": "rates", "ir_factor_2": "rates",
                },
                mapping_path=str(mapping_csv),
            ),
            "rates": AssetFactorRule(
                asset_class="rates", source="attribute", key_field="",
                factor_fields=["RatesFactor"],
            ),
        }),
        elementary=ElementaryConfig(grids={}),
        clustering=ClusteringConfig(keys=["AssetClass", "DeskName"]),
        output=OutputConfig(root=tmp_path, run_id="test_run"),
    )


def test_full_pipeline(tmp_path):
    mapping = (
        "ccy,fx_risk_factor_1,fx_risk_factor_2,ir_factor_1,ir_factor_2\n"
        "EURUSD,FX.SPOT.USD.EUR,FX.SPOT.USD.USD,IR_CURVE_SWAP.EUR,IR_CURVE_SWAP.USD\n"
        "GBPUSD,FX.SPOT.USD.GBP,FX.SPOT.USD.USD,IR_CURVE_SWAP.GBP,IR_CURVE_SWAP.USD\n"
        "AUDUSD,FX.SPOT.USD.AUD,FX.SPOT.USD.USD,IR_CURVE_SWAP.AUD,IR_CURVE_SWAP.USD\n"
    )
    mapping_csv = tmp_path / "fx_mapping.csv"
    mapping_csv.write_text(mapping)

    ctx = run(_config(tmp_path, mapping_csv), MockPortfolioClient(), MockMarketDataClient())

    assert len(ctx.clusters.clusters) > 0
    assert ctx.elementary.n_trades > 0

    run_dir = tmp_path / "test_run"
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "jobs.pkl").exists()
    assert (run_dir / "config.snapshot.yaml").exists()

    with open(run_dir / "jobs.pkl", "rb") as fh:
        jobs = pickle.load(fh)
    assert len(jobs) == len(ctx.clusters.clusters)
    for entry in jobs:
        assert entry["n_scenarios"] >= 0
        assert (run_dir / "clusters" / entry["cluster_id"]).exists()
```

### `tests/unit/__init__.py`

```python

```

### `tests/unit/test_marketdata.py`

```python
"""Unit tests for the market-data layer."""
import numpy as np
import pytest

from src.rade_static_replication.domain.errors import MarketDataError
from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.fx.instruments import VolSurface


def test_discount_curve_df_monotone():
    c = DiscountCurve("USD", np.array([0.5, 1.0, 2.0, 5.0]), np.array([0.04, 0.042, 0.044, 0.045]))
    assert c.df(0.0) == 1.0
    dfs = [c.df(t) for t in (0.5, 1.0, 2.0, 5.0)]
    assert all(dfs[i] > dfs[i + 1] for i in range(len(dfs) - 1))
    # zero/df round-trip
    assert c.df(1.0) == pytest.approx(np.exp(-c.zero(1.0) * 1.0), rel=1e-10)


def test_curve_rejects_unsorted():
    with pytest.raises(MarketDataError):
        DiscountCurve("USD", np.array([1.0, 0.5]), np.array([0.04, 0.04]))


def test_vol_surface_interp_in_range():
    exp = np.array([0.25, 1.0, 2.0])
    k = np.array([0.9, 1.0, 1.1])
    vols = np.array([[0.12, 0.10, 0.11], [0.13, 0.11, 0.12], [0.14, 0.12, 0.13]])
    s = VolSurface(exp, k, vols)
    v = s.vol(1.0, 0.5)
    assert 0.09 < v < 0.15
```

### `tests/unit/test_pricing.py`

```python
"""Unit tests for pricing kernels (sanity / parity)."""
import math

import pytest

from src.rade_static_replication.pricing.kernels.fx import price_fx_forward, price_fx_vanilla


def test_fx_put_call_parity():
    S, K, T, r_d, r_f, vol = 1.10, 1.12, 0.75, 0.045, 0.032, 0.10
    call = price_fx_vanilla(S, K, T, r_d, r_f, vol, True, 1.0)
    put = price_fx_vanilla(S, K, T, r_d, r_f, vol, False, 1.0)
    forward_pv = S * math.exp(-r_f * T) - K * math.exp(-r_d * T)
    assert call - put == pytest.approx(price_fx_forward(S, K, T, r_d, r_f, 1.0, 1.0), abs=1e-12)
    assert call - put == pytest.approx(forward_pv, abs=1e-12)


def test_fx_call_intrinsic_at_zero_vol():
    pv = price_fx_vanilla(1.20, 1.10, 0.0, 0.04, 0.03, 0.0, True, 1.0)
    assert pv == pytest.approx(0.10)
```
