from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Union

from src.marketdata.core.artifacts import load_market_dataset
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.providers.interfaces import MarketDataProvider
from src.marketdata.providers.static.config import StaticProviderConfig
from src.marketdata.providers.static.provider import StaticProvider
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.provider import SyntheticProvider


class ProviderBuildError(ValueError):
    """Raised when a provider spec is invalid or cannot be constructed."""


class DatasetLoader(Protocol):
    """Callable that loads a MarketDataset from a filesystem path."""
    def __call__(self, path: str) -> MarketDataset: ...


@dataclass(frozen=True, slots=True)
class SyntheticProviderSpec:
    """
    Specification for constructing a SyntheticProvider.

    Design
    ------
    - Keep orchestrators/examples provider-agnostic.
    - Determinism is controlled by (seed, config).
    """
    seed: int = 7
    config: SyntheticProviderConfig = field(default_factory=SyntheticProviderConfig)
    name: str = "SyntheticProvider"


@dataclass(frozen=True, slots=True)
class StaticProviderSpec:
    """
    Specification for constructing a StaticProvider.

    Exactly one source must be supplied:
      - dataset: in-memory MarketDataset, OR
      - dataset_path: artifact directory created by save_market_dataset(...)
    """
    dataset: Optional[MarketDataset] = None
    dataset_path: Optional[str] = None
    config: StaticProviderConfig = field(default_factory=StaticProviderConfig)
    name: str = "StaticProvider"


ProviderSpec = Union[SyntheticProviderSpec, StaticProviderSpec]


def build_provider(
    spec: ProviderSpec,
    *,
    dataset_loader: Optional[DatasetLoader] = None,
) -> MarketDataProvider:
    """
    Build a MarketDataProvider from an explicit provider spec.

    Parameters
    ----------
    spec:
        Provider specification (SyntheticProviderSpec or StaticProviderSpec).
    dataset_loader:
        Optional loader override for spec.dataset_path. Defaults to load_market_dataset.

    Returns
    -------
    MarketDataProvider
        A concrete provider implementing the protocol.

    Raises
    ------
    ProviderBuildError
        If the spec is invalid or cannot be constructed deterministically.
    """
    if isinstance(spec, SyntheticProviderSpec):
        return SyntheticProvider(
            seed=int(spec.seed),
            config=spec.config,
            name=str(spec.name),
        )

    if isinstance(spec, StaticProviderSpec):
        dataset = spec.dataset
        if dataset is None:
            path = (spec.dataset_path or "").strip()
            if not path:
                raise ProviderBuildError(
                    "StaticProviderSpec requires either:\n"
                    "  - dataset (in-memory MarketDataset), OR\n"
                    "  - dataset_path (artifact directory path)."
                )
            loader = load_market_dataset if dataset_loader is None else dataset_loader
            dataset = loader(path)

        return StaticProvider(
            dataset=dataset,
            config=spec.config,
            name=str(spec.name),
        )

    # Defensive: should be unreachable due to ProviderSpec union.
    raise ProviderBuildError(f"Unsupported provider spec type: {type(spec).__name__}")