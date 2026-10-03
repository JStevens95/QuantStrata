"""
Sage portfolio client (WIRING STUB — your work API).

Implement :meth:`load` to call the Sage trade API and return a :class:`RawPortfolio`
whose ``attributes`` is the exploded export and whose ``target_pnl`` is the scenario
PnL frame. Everything downstream already handles the exploded shape, so you only need
to reproduce those two frames.
"""
from __future__ import annotations

from src.rade_static_replication.domain.contracts import RawPortfolio


class SagePortfolioClient:
    """Adapter onto the internal Sage trade API."""

    def __init__(self, connection=None, **options) -> None:
        self.connection = connection
        self.options = options

    def load(self, cob_date: str) -> RawPortfolio:
        # attributes = self.connection.query_trade_attributes(cob_date)   # exploded DataFrame
        # target_pnl = self.connection.query_scenario_pnl(cob_date)       # trade_id x scenarios
        # return RawPortfolio(attributes=attributes, target_pnl=target_pnl,
        #                     cob_date=cob_date, source="sage")
        raise NotImplementedError("TODO(wire): call Sage and return RawPortfolio")
