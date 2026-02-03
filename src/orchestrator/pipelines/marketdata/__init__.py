"""
Market Data Pipelines Package.

Contains pipelines for market data acquisition and transformation:
- build_timeseries: Build timeseries dataset from synthetic or external providers
- replay_static: Replay static dataset from file/artifacts
- build_curves: Bootstrap yield curves from rate quotes
- build_vol_surface: Build volatility surface from option quotes
"""
from __future__ import annotations
