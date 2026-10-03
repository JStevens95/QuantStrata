"""
Mock data generation for rade_sr.

Produces *raw-shaped* portfolio attribute and scenario-PnL files that mirror the
external risk-system exports, so the whole preprocessing pipeline (ingestion →
assets → elementary trades → PnL → clusters) can be built and tested offline,
before any real market-data API is wired.

The attribute file is intentionally emitted in its raw, un-aggregated form:
one row per ``(trade, risk_type, curve)``, with ``AssetClass == "ALL"`` on the
``Notional`` rows — exactly like the real export. The matching ingestion logic
lives in ``rade_sr.replication.ingest``.

Entry point: :func:`rade_sr.mock.portfolio.generate_mock_portfolio`.
"""
from __future__ import annotations

from .portfolio import generate_mock_portfolio

__all__ = ["generate_mock_portfolio"]
