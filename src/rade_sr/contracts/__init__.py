"""
Conformance harness for the rade_sr swap seams.

Run these checks in your work environment against your real implementations to
verify each contract *before* running the full pipeline — so wiring is a pass/fail
checklist, not a debugging session.

    from src.rade_sr.contracts import run_all
    run_all(
        market_data_client=my_client,     # MarketDataClient
        portfolio_source=my_source,        # PortfolioSource (optional)
        pricer=my_pricer,                  # batch pricer (optional)
        fx_pair="EURUSD",
    )
"""
from __future__ import annotations

from .conformance import (
    CheckResult,
    check_artifacts_roundtrip,
    check_market_data_client,
    check_portfolio_source,
    check_pricer,
    run_all,
)

__all__ = [
    "CheckResult",
    "check_market_data_client",
    "check_portfolio_source",
    "check_pricer",
    "check_artifacts_roundtrip",
    "run_all",
]
