"""
Pipeline discovery / registration.

Design goals
------------
- Keep this file *boringly simple* and easy to read.
- Avoid import side-effects elsewhere by importing pipeline modules here only.
- Keep pipeline names Vn-proof by registering a *versioned* name.
- Optionally provide an *alias* for convenience in examples.

Add new pipelines
-----------------
1) Import the new pipeline builder inside `register_builtin_pipelines`.
2) Register it under a versioned name: "<domain>.<pipeline>.vN".
3) (Optional) Register an alias without ".vN" pointing to the current default.

Pipeline Categories
-------------------
- marketdata: Market data acquisition and transformation
- portfolio: Portfolio construction and management
- pricing: Pricing operations
- risk: Risk analytics (scenarios, sensitivities, VaR, attribution)
- calibration: Model calibration (vol surface, Heston, Hull-White)
- ml: Machine learning training pipelines
- backtest: Backtesting and model validation
- workflow: End-to-end composite workflows
"""

from __future__ import annotations

from src.orchestrator.core.registry import PipelineRegistry


def register_builtin_pipelines(registry: PipelineRegistry) -> None:
    """
    Register all built-in pipelines into the provided registry.

    Parameters
    ----------
    registry:
        Registry instance that maps pipeline names -> builder functions.

    Notes
    -----
    - We import pipeline builders inside this function to avoid import-time side effects.
    - Prefer *versioned* names as the stable configuration contract (Vn-proof).
    """

    # =========================================================================
    # MARKETDATA PIPELINES
    # =========================================================================
    
    # build_timeseries: Build timeseries dataset from synthetic or external providers
    from src.orchestrator.pipelines.marketdata.build_timeseries import (
        build_pipeline as md_build_timeseries,
    )
    registry.register("marketdata.build_timeseries", md_build_timeseries)

    # replay_static: Replay static dataset from file/artifacts
    from src.orchestrator.pipelines.marketdata.replay_static import (
        build_pipeline as md_replay_static,
    )
    registry.register("marketdata.replay_static", md_replay_static)

    # build_curves: Bootstrap yield curves from rate quotes
    from src.orchestrator.pipelines.marketdata.build_curves import (
        build_pipeline as md_build_curves,
    )
    registry.register("marketdata.build_curves", md_build_curves)

    # build_vol_surface: Build volatility surface from option quotes
    from src.orchestrator.pipelines.marketdata.build_vol_surface import (
        build_pipeline as md_build_vol_surface,
    )
    registry.register("marketdata.build_vol_surface", md_build_vol_surface)

    # =========================================================================
    # PORTFOLIO PIPELINES
    # =========================================================================

    # build_from_config: Build portfolio from YAML/JSON position specification
    from src.orchestrator.pipelines.portfolio.build_from_config import (
        build_pipeline as pf_build_from_config,
    )
    registry.register("portfolio.build_from_config", pf_build_from_config)

    # construct_hedge: Construct hedge portfolio to neutralise Greek exposures
    from src.orchestrator.pipelines.portfolio.construct_hedge import (
        build_pipeline as pf_construct_hedge,
    )
    registry.register("portfolio.construct_hedge", pf_construct_hedge)

    # =========================================================================
    # PRICING PIPELINES
    # =========================================================================

    # price_portfolio: Price a portfolio using the pricer registry
    from src.orchestrator.pipelines.pricing.price_portfolio import (
        build_pipeline as px_price_portfolio,
    )
    registry.register("pricing.price_portfolio", px_price_portfolio)

    # =========================================================================
    # RISK PIPELINES
    # =========================================================================

    # run_scenarios: Run scenario analysis (spot/vol/rate shocks)
    from src.orchestrator.pipelines.risk.run_scenarios import (
        build_pipeline as rsk_run_scenarios,
    )
    registry.register("risk.run_scenarios", rsk_run_scenarios)

    # compute_sensitivities: Compute portfolio Greeks with aggregation
    from src.orchestrator.pipelines.risk.compute_sensitivities import (
        build_pipeline as rsk_compute_sensitivities,
    )
    registry.register("risk.compute_sensitivities", rsk_compute_sensitivities)

    # compute_var: Compute Value-at-Risk using multiple methods
    from src.orchestrator.pipelines.risk.compute_var import (
        build_pipeline as rsk_compute_var,
    )
    registry.register("risk.compute_var", rsk_compute_var)

    # pnl_attribution: Attribute P&L changes to risk factors
    from src.orchestrator.pipelines.risk.pnl_attribution import (
        build_pipeline as rsk_pnl_attribution,
    )
    registry.register("risk.pnl_attribution", rsk_pnl_attribution)

    # validate_greeks: Validate analytic Greeks against bump-and-reprice
    from src.orchestrator.pipelines.risk.validate_greeks import (
        build_pipeline as rsk_validate_greeks,
    )
    registry.register("risk.validate_greeks", rsk_validate_greeks)

    # =========================================================================
    # CALIBRATION PIPELINES
    # =========================================================================

    # volatility_surface: Calibrate vol surface (SABR, Dupire, SVI)
    from src.orchestrator.pipelines.calibration.volatility_surface import (
        build_pipeline as cal_volatility_surface,
    )
    registry.register("calibration.volatility_surface", cal_volatility_surface)

    # stochastic_vol: Calibrate Heston stochastic volatility model
    from src.orchestrator.pipelines.calibration.stochastic_vol import (
        build_pipeline as cal_stochastic_vol,
    )
    registry.register("calibration.stochastic_vol", cal_stochastic_vol)

    # short_rate: Calibrate Hull-White short rate model
    from src.orchestrator.pipelines.calibration.short_rate import (
        build_pipeline as cal_short_rate,
    )
    registry.register("calibration.short_rate", cal_short_rate)

    # =========================================================================
    # MACHINE LEARNING PIPELINES
    # =========================================================================

    # train_deep_hedging: Train deep hedging agent
    from src.orchestrator.pipelines.ml.train_deep_hedging import (
        build_pipeline as ml_train_deep_hedging,
    )
    registry.register("ml.train_deep_hedging", ml_train_deep_hedging)

    # train_gnn_pricer: Train GNN-RNN hybrid pricing model
    from src.orchestrator.pipelines.ml.train_gnn_pricer import (
        build_pipeline as ml_train_gnn_pricer,
    )
    registry.register("ml.train_gnn_pricer", ml_train_gnn_pricer)

    # train_calibration_model: Train ML calibration accelerator
    from src.orchestrator.pipelines.ml.train_calibration_model import (
        build_pipeline as ml_train_calibration_model,
    )
    registry.register("ml.train_calibration_model", ml_train_calibration_model)

    # =========================================================================
    # BACKTESTING PIPELINES
    # =========================================================================

    # run_strategy: Run trading strategy backtest
    from src.orchestrator.pipelines.backtest.run_strategy import (
        build_pipeline as bt_run_strategy,
    )
    registry.register("backtest.run_strategy", bt_run_strategy)

    # model_comparison: Compare pricing results across models
    from src.orchestrator.pipelines.backtest.model_comparison import (
        build_pipeline as bt_model_comparison,
    )
    registry.register("backtest.model_comparison", bt_model_comparison)

    # =========================================================================
    # WORKFLOW PIPELINES (End-to-End)
    # =========================================================================

    # options_desk_daily: Complete daily workflow for options desk
    from src.orchestrator.pipelines.workflow.options_desk_daily import (
        build_pipeline as wf_options_desk_daily,
    )
    registry.register("workflow.options_desk_daily", wf_options_desk_daily)

    # trade_lifecycle: Trade lifecycle management
    from src.orchestrator.pipelines.workflow.trade_lifecycle import (
        build_pipeline as wf_trade_lifecycle,
    )
    registry.register("workflow.trade_lifecycle", wf_trade_lifecycle)

    # hedging_simulation: Hedging strategy simulation
    from src.orchestrator.pipelines.workflow.hedging_simulation import (
        build_pipeline as wf_hedging_simulation,
    )
    registry.register("workflow.hedging_simulation", wf_hedging_simulation)
