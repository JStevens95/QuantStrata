# QuantStrata Notebook & Example Audit Report

## Summary

- **Total files tested:** 82
- **Passed:** 74
- **Failed:** 8
- **Skipped:** 0
- **Pass rate:** 90.2%

## Failures

### `docs/tutorials/machine_learning/ml_model_lifecycle.ipynb`
- **Category:** syntax
- **Error type:** SyntaxError
- **Line:** 46
- **Error:** `Cell 1: invalid syntax`

### `docs/tutorials/machine_learning/ml_production.ipynb`
- **Category:** import
- **Error type:** TimeoutError
- **Error:** `Import check timed out (60s)`

### `docs/tutorials/deep_hedging/deep_hedging_tutorial.ipynb`
- **Category:** import
- **Error type:** TypeError
- **Error:** `TypeError: dataclass() got an unexpected keyword argument 'slots'`

### `docs/tutorials/q_learning/rl_deployment_tutorial.ipynb`
- **Category:** import
- **Error type:** TypeError
- **Error:** `TypeError: dataclass() got an unexpected keyword argument 'slots'`

### `docs/tutorials/pricing/exotic_options.ipynb`
- **Category:** import
- **Error type:** TypeError
- **Error:** `TypeError: dataclass() got an unexpected keyword argument 'slots'`

### `examples/showcase/01_european_vanilla_pricing.py`
- **Category:** import
- **Error type:** ModuleNotFoundError
- **Error:** `ModuleNotFoundError: No module named 'src'`

### `examples/fundamentals/01_market_ids_and_quotes.py`
- **Category:** import
- **Error type:** ModuleNotFoundError
- **Error:** `ModuleNotFoundError: No module named 'src'`

### `examples/pipelines/run_build_curves.py`
- **Category:** import
- **Error type:** ModuleNotFoundError
- **Error:** `ModuleNotFoundError: No module named 'src'`

## Passed Files

- `docs/tutorials/deep_hedging/deep_hedging_tutorial.ipynb` (syntax)
- `docs/tutorials/backtesting/backtesting_introduction.ipynb` (syntax)
- `docs/tutorials/market-data/synthetic_data_generation.ipynb` (syntax)
- `docs/tutorials/market-data/ir_volatility_surfaces.ipynb` (syntax)
- `docs/tutorials/calibration/stochastic_vol_heston_analysis.ipynb` (syntax)
- `docs/tutorials/calibration/calibration_framework.ipynb` (syntax)
- `docs/tutorials/calibration/calibration_curve_bootstrapping.ipynb` (syntax)
- `docs/tutorials/calibration/local_volatility_analysis.ipynb` (syntax)
- `docs/tutorials/calibration/calibration_volatility_surface.ipynb` (syntax)
- `docs/tutorials/machine_learning/hybrid_gnn_lstm_tutorial.ipynb` (syntax)
- `docs/tutorials/machine_learning/ml_production.ipynb` (syntax)
- `docs/tutorials/q_learning/rl_deployment_tutorial.ipynb` (syntax)
- `docs/tutorials/models/neural_sde_tutorial.ipynb` (syntax)
- `docs/tutorials/streaming/streaming_and_live_data.ipynb` (syntax)
- `docs/tutorials/instruments/lookback_options_analysis.ipynb` (syntax)
- `docs/tutorials/instruments/forward_options.ipynb` (syntax)
- `docs/tutorials/instruments/digital_options_analysis.ipynb` (syntax)
- `docs/tutorials/instruments/asian_options_analysis.ipynb` (syntax)
- `docs/tutorials/instruments/vanilla_options_analysis.ipynb` (syntax)
- `docs/tutorials/instruments/barrier_options_analysis.ipynb` (syntax)
- `docs/tutorials/instruments/futures_options.ipynb` (syntax)
- `docs/tutorials/instruments/double_barrier_options_analysis.ipynb` (syntax)
- `docs/tutorials/instruments/touch_options_analysis.ipynb` (syntax)
- `docs/tutorials/performance/performance_and_scalability.ipynb` (syntax)
- `docs/tutorials/risk/risk_introduction.ipynb` (syntax)
- `docs/tutorials/pricing/fx_options_pricing.ipynb` (syntax)
- `docs/tutorials/pricing/lmm_pricing.ipynb` (syntax)
- `docs/tutorials/pricing/sabr_model.ipynb` (syntax)
- `docs/tutorials/pricing/ir_instruments_pricing.ipynb` (syntax)
- `docs/tutorials/pricing/equity_options_pricing.ipynb` (syntax)
- `docs/tutorials/pricing/exotic_options.ipynb` (syntax)
- `docs/tutorials/pricing/multi_asset_options.ipynb` (syntax)
- `docs/tutorials/pricing/advanced_mc_methods.ipynb` (syntax)
- `docs/tutorials/pricing/jump_levy_models.ipynb` (syntax)
- `docs/tutorials/pricing/bond_pricing.ipynb` (syntax)
- `docs/tutorials/analytics/advanced_analytics_reporting.ipynb` (syntax)
- `examples/notebooks/04_multi_pipeline_workflow.ipynb` (syntax)
- `examples/notebooks/01_pricing_and_greeks.ipynb` (syntax)
- `examples/notebooks/02_volatility_surfaces.ipynb` (syntax)
- `examples/notebooks/05_deep_hedging.ipynb` (syntax)
- `examples/notebooks/03_scenario_analysis.ipynb` (syntax)
- `examples/fundamentals/06_scenario_shocks.py` (syntax)
- `examples/fundamentals/03_volatility_surface.py` (syntax)
- `examples/fundamentals/05_market_snapshot.py` (syntax)
- `examples/fundamentals/02_curves_and_term_structures.py` (syntax)
- `examples/fundamentals/04_timeseries_datasets.py` (syntax)
- `examples/fundamentals/01_market_ids_and_quotes.py` (syntax)
- `examples/workflows/calibration_to_pricing.py` (syntax)
- `examples/workflows/options_desk_daily.py` (syntax)
- `examples/pipelines/run_build_curves.py` (syntax)
- `examples/pipelines/run_backtest_strategy.py` (syntax)
- `examples/pipelines/run_var.py` (syntax)
- `examples/pipelines/run_train_neural_sde.py` (syntax)
- `examples/pipelines/run_deploy_rl_agent.py` (syntax)
- `examples/pipelines/run_hyperparameter_tuning.py` (syntax)
- `examples/pipelines/run_train_gnn_pricer.py` (syntax)
- `examples/pipelines/run_backtest_hedging_agent.py` (syntax)
- `examples/pipelines/run_compute_greeks.py` (syntax)
- `examples/pipelines/run_build_vol_surface.py` (syntax)
- `examples/pipelines/run_calibrate_sabr.py` (syntax)
- `examples/pipelines/run_portfolio_optimisation.py` (syntax)
- `examples/pipelines/run_portfolio_from_config.py` (syntax)
- `examples/pipelines/run_calibrate_heston.py` (syntax)
- `examples/showcase/03_advanced_models.py` (syntax)
- `examples/showcase/02_exotic_options_gallery.py` (syntax)
- `examples/showcase/01_european_vanilla_pricing.py` (syntax)
- `examples/risk/02_sensitivities_computation.py` (syntax)
- `examples/risk/01_scenario_analysis.py` (syntax)
- `examples/pricing/03_portfolio_pricing.py` (syntax)
- `examples/pricing/01_equity_vanilla_pricing.py` (syntax)
- `examples/pricing/02_exotic_options.py` (syntax)
- `examples/pricing/01_fx_vanilla_pricing.py` (syntax)
- `examples/pricing/exotic_structured_products.py` (syntax)
- `docs/tutorials/models/neural_sde_tutorial.ipynb` (import)