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
