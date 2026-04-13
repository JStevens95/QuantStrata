"""
Application constants for the Ensemble Analytics dashboard.

Centralises tab IDs, split names, default ports, and display strings
so that layout modules and callbacks reference a single source of truth.
"""
from typing import Dict, List, Tuple

# ── App metadata ──────────────────────────────────────────────────
APP_TITLE: str = "Ensemble Analytics — Hybrid GNN-RNN"
DEFAULT_PORT: int = 8051

# ── Tab identifiers (must match dcc.Tab value attributes) ─────────
TAB_OVERVIEW: str = "tab-overview"
TAB_EVALUATION: str = "tab-evaluation"
TAB_CLUSTER_DEEP_DIVE: str = "tab-cluster-deep-dive"
TAB_MARKET_DATA: str = "tab-market-data"
TAB_TRADE_GRAPH: str = "tab-trade-graph"
TAB_INFERENCE: str = "tab-inference"
TAB_GOVERNANCE: str = "tab-governance"

TAB_ORDER: List[Tuple[str, str]] = [
    (TAB_OVERVIEW, "Overview"),
    (TAB_EVALUATION, "Evaluation"),
    (TAB_CLUSTER_DEEP_DIVE, "Cluster Deep Dive"),
    (TAB_MARKET_DATA, "Market Data"),
    (TAB_TRADE_GRAPH, "Trade Graph"),
    (TAB_INFERENCE, "Inference"),
    (TAB_GOVERNANCE, "Model Governance"),
]

# ── Evaluation sub-tab identifiers ────────────────────────────────
EVAL_SUB_PORTFOLIO: str = "eval-sub-portfolio"
EVAL_SUB_DESK: str = "eval-sub-desk"
EVAL_SUB_PRODUCT: str = "eval-sub-product"
EVAL_SUB_CCY: str = "eval-sub-ccy"
EVAL_SUB_CLUSTER: str = "eval-sub-cluster"

EVAL_SUB_ORDER: List[Tuple[str, str]] = [
    (EVAL_SUB_PORTFOLIO, "Portfolio"),
    (EVAL_SUB_DESK, "By Desk"),
    (EVAL_SUB_PRODUCT, "By Product"),
    (EVAL_SUB_CCY, "By CCY"),
    (EVAL_SUB_CLUSTER, "By Cluster"),
]

# ── Market Data sub-tab identifiers ──────────────────────────────
MD_SUB_RF_SUMMARY: str = "md-sub-rf-summary"
MD_SUB_SHOCK_EXPLORER: str = "md-sub-shock-explorer"
MD_SUB_SCENARIO_HEATMAP: str = "md-sub-scenario-heatmap"
MD_SUB_DISTRIBUTION: str = "md-sub-distribution"

MD_SUB_ORDER: List[Tuple[str, str]] = [
    (MD_SUB_RF_SUMMARY, "RF Summary"),
    (MD_SUB_SHOCK_EXPLORER, "Shock Explorer"),
    (MD_SUB_SCENARIO_HEATMAP, "Scenario Heatmap"),
    (MD_SUB_DISTRIBUTION, "Distribution"),
]

# ── Trade Graph sub-tab identifiers ──────────────────────────────
TG_SUB_GRAPH_VIEW: str = "tg-sub-graph-view"
TG_SUB_ADJACENCY: str = "tg-sub-adjacency"
TG_SUB_NODE_ANALYTICS: str = "tg-sub-node-analytics"
TG_SUB_CROSS_CLUSTER: str = "tg-sub-cross-cluster"

TG_SUB_ORDER: List[Tuple[str, str]] = [
    (TG_SUB_GRAPH_VIEW, "Graph View"),
    (TG_SUB_ADJACENCY, "Adjacency Analysis"),
    (TG_SUB_NODE_ANALYTICS, "Node Analytics"),
    (TG_SUB_CROSS_CLUSTER, "Cross-Cluster"),
]

# ── Split names ───────────────────────────────────────────────────
SPLITS: List[str] = ["test", "val", "train"]
DEFAULT_SPLIT: str = "test"

# ── Evaluation group-by column mapping ────────────────────────────
# Maps the internal sub-tab concept to the actual catalogue column name.
# Update these values to match your trade catalogue columns.
EVAL_GROUP_COLUMNS: Dict[str, str] = {
    "desk": "AssetClassCode",
    "product_type": "ProductCode",
    "ccy": "CurrencyCode",
}

# ── Display format helpers ────────────────────────────────────────
METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "mae": "MAE",
    "rmse": "RMSE",
    "max_ae": "Max AE",
    "mape": "MAPE",
    "r2": "R²",
    "p95_ae": "P95 AE",
    "p99_ae": "P99 AE",
}
