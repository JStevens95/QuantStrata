# src/orchestrator/core/state_keys.py
from __future__ import annotations

class StateKeys:
    # ---- marketdata outputs ----
    MARKET_ID_STRINGS = "market_id_strings"
    MARKET_IDS = "market_ids"
    MARKET_IDS_PRETTY = "market_ids_pretty"
    UNIVERSE = "universe"
    REQUEST = "request"
    REQUEST_SUMMARY = "request_summary"
    DATASET = "dataset"
    MARKET = "market"
    MARKET_SNAPSHOT_SUMMARY = "market_snapshot_summary"

    # ---- pricing outputs ----
    PRICER_REGISTRY = "pricer_registry"
    PORTFOLIO = "portfolio"
    PORTFOLIO_PRICING_RESULT = "portfolio_pricing_result"
    PORTFOLIO_PRICING_SUMMARY = "portfolio_pricing_summary"
    PRICING_SUMMARY_JSON_PATH = "pricing_summary_json_path"
    PORTFOLIO_PRICING_CSV_PATH = "portfolio_pricing_csv_path"