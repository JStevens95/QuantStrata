"""
Portfolio-source seam.

``PortfolioSource`` is the contract for obtaining the two *raw* portfolio frames
(attributes + scenario PnL) that Step 0 ingestion consumes. Swap mock for real by
implementing two methods.

Implementations
---------------
- :class:`MockPortfolioSource` — generates a synthetic FX desk book in memory.
- :class:`FilePortfolioSource`  — reads CSV/Parquet exported from your systems.
- :class:`ApiPortfolioSource`   — wiring stub for a live portfolio API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, Tuple, Union, runtime_checkable

import pandas as pd


@runtime_checkable
class PortfolioSource(Protocol):
    """Returns the raw attribute and scenario-PnL frames (un-normalized)."""

    def load_attributes(self) -> pd.DataFrame:
        """Raw attribute frame: one row per (trade, risk_type, curve)."""
        ...

    def load_pnl(self) -> pd.DataFrame:
        """Raw PnL frame: trade-id index, scenario-date columns, + AssetClass."""
        ...


class MockPortfolioSource:
    """In-memory synthetic FX desk portfolio (no I/O)."""

    def __init__(
        self,
        n_trades: int = 2500,
        n_scenarios: int = 250,
        cob_date: str = "2026-05-29",
        seed: int = 42,
    ) -> None:
        from src.rade_sr.mock.portfolio import generate_mock_portfolio

        self._attributes, self._pnl = generate_mock_portfolio(
            n_trades=n_trades,
            n_scenarios=n_scenarios,
            cob_date=cob_date,
            seed=seed,
        )

    def load_attributes(self) -> pd.DataFrame:
        return self._attributes.copy()

    def load_pnl(self) -> pd.DataFrame:
        return self._pnl.copy()


class FilePortfolioSource:
    """Reads the raw frames from CSV or Parquet files on disk.

    Parameters
    ----------
    attributes_path, pnl_path : str or Path
        Paths to the raw attribute and PnL exports.
    pnl_index_col : str
        Index column for the PnL file (the trade id).
    """

    def __init__(
        self,
        attributes_path: Union[str, Path],
        pnl_path: Union[str, Path],
        pnl_index_col: str = "PTSDealNumber",
    ) -> None:
        self._attr_path = Path(attributes_path)
        self._pnl_path = Path(pnl_path)
        self._pnl_index_col = pnl_index_col

    @staticmethod
    def _read(path: Path, **kwargs: Any) -> pd.DataFrame:
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path, **kwargs)

    def load_attributes(self) -> pd.DataFrame:
        return self._read(self._attr_path)

    def load_pnl(self) -> pd.DataFrame:
        df = self._read(self._pnl_path)
        if self._pnl_index_col in df.columns:
            df = df.set_index(self._pnl_index_col)
        df.index = df.index.astype(str)
        return df


class ApiPortfolioSource:
    """Wiring stub for a live portfolio API.

    Parameters
    ----------
    client : Any
        Your portfolio API client.
    portfolio_id, as_of : str
        Selection parameters for the export.

    TODO(wire): Replace the method bodies with your API calls that return the
    raw attribute and scenario-PnL frames in the documented schema.
    """

    def __init__(self, client: Any, portfolio_id: str, as_of: Optional[str] = None) -> None:
        self._client = client
        self._portfolio_id = portfolio_id
        self._as_of = as_of

    def load_attributes(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Wire ApiPortfolioSource.load_attributes() to your portfolio API"
        )

    def load_pnl(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Wire ApiPortfolioSource.load_pnl() to your scenario-PnL API"
        )


# ─────────────────────────────────────────────────────────────────────────
# Mode selection — the single switch between CSV / API / mock
# ─────────────────────────────────────────────────────────────────────────

_SOURCES = {
    "csv": FilePortfolioSource,
    "file": FilePortfolioSource,
    "parquet": FilePortfolioSource,
    "api": ApiPortfolioSource,
    "mock": MockPortfolioSource,
}


def make_portfolio_source(mode: str, **kwargs: Any) -> PortfolioSource:
    """Build the right :class:`PortfolioSource` for a mode. The one switch point.

    Keeps the orchestrator mode-agnostic: pick the source here, pass the result in.

    Examples
    --------
    CSV mode::

        src = make_portfolio_source(
            "csv", attributes_path="attrs.csv", pnl_path="pnl.csv",
        )

    API mode::

        src = make_portfolio_source(
            "api", client=my_api_client, portfolio_id="FX_BOOK",
        )

    Parameters
    ----------
    mode : str
        One of ``"csv"``/``"file"``/``"parquet"``, ``"api"``, ``"mock"``.
    **kwargs
        Constructor arguments for the chosen source class.
    """
    key = mode.lower()
    if key not in _SOURCES:
        raise ValueError(
            f"Unknown portfolio source mode {mode!r}; choose from {sorted(_SOURCES)}"
        )
    return _SOURCES[key](**kwargs)
