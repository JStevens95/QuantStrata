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
