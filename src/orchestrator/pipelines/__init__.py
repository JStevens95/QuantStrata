"""
Pipeline builders package.

A pipeline builder:
- Reads RunConfig
- Returns a Pipeline made of Steps

Pipeline Categories
-------------------
marketdata/
    - build_timeseries: Build timeseries dataset from synthetic/external providers
    - replay_static: Replay static dataset from file/artifacts
    - build_curves: Bootstrap yield curves from rate quotes
    - build_vol_surface: Build volatility surface from option quotes

portfolio/
    - build_from_config: Build portfolio from YAML/JSON position specification
    - construct_hedge: Construct hedge portfolio to neutralise Greek exposures

pricing/
    - price_portfolio: Price portfolio using pricer registry

risk/
    - run_scenarios: Run scenario analysis (spot/vol/rate shocks)
    - compute_sensitivities: Compute portfolio Greeks with aggregation
    - compute_var: Compute Value-at-Risk using multiple methods
    - pnl_attribution: Attribute P&L to risk factors
    - validate_greeks: Validate analytic Greeks against bump-and-reprice

calibration/
    - volatility_surface: Calibrate vol surface (SABR, Dupire, SVI)
    - stochastic_vol: Calibrate Heston stochastic volatility model
    - short_rate: Calibrate Hull-White short rate model

ml/
    - train_deep_hedging: Train deep hedging agent
    - train_gnn_pricer: Train GNN-RNN hybrid pricing model
    - train_calibration_model: Train ML calibration accelerator

backtest/
    - run_strategy: Run trading strategy backtest
    - model_comparison: Compare pricing results across models

workflow/
    - options_desk_daily: Complete daily workflow for options desk
    - trade_lifecycle: Trade lifecycle management
    - hedging_simulation: Hedging strategy simulation

Usage
-----
Pipelines are registered in `src.orchestrator.runtime.discovery` and
executed via `run_pipeline_from_config()` in entrypoints.

Example:
    from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
    from src.orchestrator.config.schemas import RunConfig
    
    cfg = RunConfig(
        pipeline="workflow.options_desk_daily",
        io={"artifacts_dir": "./artifacts"},
        params={"workflow": {}},
    )
    ctx = run_pipeline_from_config(cfg)
"""
from __future__ import annotations
