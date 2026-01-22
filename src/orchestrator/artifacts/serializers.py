"""
Domain serializer wrappers.

Why wrappers?
-------------
The orchestrator should not be tightly coupled to where domain serialization lives.
These thin wrappers provide a stable seam for Vn refactors.

Current domain dependency:
- marketdata dataset save/load functions
"""

from __future__ import annotations

from pathlib import Path

from src.marketdata.core.artifacts import load_market_dataset, save_market_dataset
from src.marketdata.core.dataset import MarketDataset


def save_dataset(dataset: MarketDataset, path: str | Path, *, overwrite: bool = False) -> Path:
    """
    Save a MarketDataset to an artifact directory.
    """
    return save_market_dataset(dataset, path, overwrite=overwrite)


def load_dataset(path: str | Path) -> MarketDataset:
    """
    Load a MarketDataset from an artifact directory.
    """
    return load_market_dataset(path)