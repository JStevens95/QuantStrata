# src/orchestrator/core/state_keys.py
"""
Centralised state key constants for the orchestrator framework.

Design Philosophy
-----------------
- All state keys are defined here to ensure consistency across pipelines.
- Keys are grouped by domain (marketdata, pricing, risk, calibration, ML, backtest).
- Use these constants instead of string literals in pipeline steps.
- Keeps the pipeline code type-safe and refactor-friendly.

Usage
-----
    from src.orchestrator.core.state_keys import StateKeys as Keys
    
    ctx.put(Keys.MARKET, market_snapshot)
    market = ctx.get(Keys.MARKET)
"""
from __future__ import annotations


class StateKeys:
    """
    Centralised state key constants for ctx.state dictionary.
    
    Organised by domain:
    - Market Data: Quotes, curves, surfaces, datasets
    - Portfolio: Positions, instruments, portfolio objects
    - Pricing: Pricer registry, pricing results
    - Risk: Scenarios, sensitivities, VaR, attribution
    - Calibration: Parameters, calibrated objects
    - Machine Learning: Models, training results
    - Backtesting: Strategy results, performance metrics
    - Workflow: Composite workflow outputs
    """
    
    # =========================================================================
    # MARKET DATA KEYS
    # =========================================================================
    
    # --- Market identifiers ---
    MARKET_ID_STRINGS = "market_id_strings"      # List[str]: Raw market ID strings
    MARKET_IDS = "market_ids"                    # List[MarketId]: Parsed market IDs
    MARKET_IDS_PRETTY = "market_ids_pretty"      # List[str]: Pretty-printed IDs
    UNIVERSE = "universe"                        # Universe: Collection of market IDs
    
    # --- Market data requests ---
    REQUEST = "request"                          # TimeseriesRequest: Data request
    REQUEST_SUMMARY = "request_summary"          # Dict: Request summary for logging
    
    # --- Market data objects ---
    DATASET = "dataset"                          # MarketDataset: Timeseries dataset
    MARKET = "market"                            # Market: Point-in-time snapshot
    MARKET_SNAPSHOT_SUMMARY = "market_snapshot_summary"  # Dict: Snapshot metadata
    START_MARKET = "start_market"                # Market: T-1 market (for attribution)
    END_MARKET = "end_market"                    # Market: T market (for attribution)
    
    # --- Rate quotes and curves ---
    RATE_QUOTES = "rate_quotes"                  # List[RateQuote]: Input rate quotes
    TERM_STRUCTURE = "term_structure"            # TermStructure: Bootstrapped curve
    DISCOUNT_FACTORS = "discount_factors"        # Dict[str, float]: DF by tenor
    
    # --- Volatility quotes and surfaces ---
    VOL_QUOTES = "vol_quotes"                    # List[VolQuote]: Input vol quotes
    VOL_SURFACE = "vol_surface"                  # VolSurface: Calibrated surface
    ARBITRAGE_REPORT = "arbitrage_report"        # Dict: Arbitrage validation results
    
    # =========================================================================
    # PORTFOLIO KEYS
    # =========================================================================
    
    POSITION_CONFIGS = "position_configs"        # List[Dict]: Parsed position configs
    INSTRUMENTS = "instruments"                  # List[Instrument]: Instantiated instruments
    POSITIONS = "positions"                      # List[Position]: Position objects
    PORTFOLIO = "portfolio"                      # Portfolio: Assembled portfolio
    
    # --- Hedging ---
    PORTFOLIO_GREEKS = "portfolio_greeks"        # Dict[str, float]: Current Greek exposures
    TARGET_GREEKS = "target_greeks"              # Dict[str, float]: Target exposures
    HEDGE_QUANTITIES = "hedge_quantities"        # Dict[str, float]: Optimal hedge quantities
    HEDGE_PORTFOLIO = "hedge_portfolio"          # Portfolio: Constructed hedge
    RESIDUAL_GREEKS = "residual_greeks"          # Dict[str, float]: Greeks after hedging
    HEDGE_INSTRUMENTS = "hedge_instruments"      # List[Instrument]: Available hedge instruments
    
    # =========================================================================
    # PRICING KEYS
    # =========================================================================
    
    PRICER_REGISTRY = "pricer_registry"          # PricerRegistry: Pricer lookup
    PORTFOLIO_PRICING_RESULT = "portfolio_pricing_result"  # PortfolioResult: Full result
    PORTFOLIO_PRICING_SUMMARY = "portfolio_pricing_summary"  # Dict: Summary stats
    PRICING_SUMMARY_JSON_PATH = "pricing_summary_json_path"  # str: Artifact path
    PORTFOLIO_PRICING_CSV_PATH = "portfolio_pricing_csv_path"  # str: CSV artifact path
    
    # =========================================================================
    # RISK KEYS
    # =========================================================================
    
    # --- Scenarios ---
    SCENARIO_PACK = "scenario_pack"              # List[Tuple[str, Shock]]: Scenario defs
    SCENARIO_RESULT = "scenario_result"          # ScenarioResult: Full scenario results
    SCENARIO_REPORT = "scenario_report"          # ScenarioReport: Formatted report
    
    # --- Sensitivities (Greeks) ---
    SENSITIVITIES_CONFIG = "sensitivities_config"  # Dict: Greeks config
    POSITION_GREEKS = "position_greeks"          # Dict[str, Greeks]: Greeks per position
    AGGREGATED_GREEKS = "aggregated_greeks"      # Dict[str, Dict]: Aggregated by dimension
    CROSS_GAMMA_MATRIX = "cross_gamma_matrix"    # DataFrame: Cross-gamma matrix
    SENSITIVITIES_REPORT = "sensitivities_report"  # SensitivitiesReport: Full report
    
    # --- Value-at-Risk ---
    HISTORICAL_RETURNS = "historical_returns"    # DataFrame: Historical return data
    HISTORICAL_VAR = "historical_var"            # Dict[float, float]: VaR by confidence
    PARAMETRIC_VAR = "parametric_var"            # Dict[float, float]: Parametric VaR
    MONTE_CARLO_VAR = "monte_carlo_var"          # Dict[float, float]: MC VaR
    EXPECTED_SHORTFALL = "expected_shortfall"    # Dict[str, Dict]: ES by method
    VAR_DECOMPOSITION = "var_decomposition"      # Dict[str, float]: VaR by risk factor
    VAR_REPORT = "var_report"                    # VaRReport: Full VaR report
    
    # --- P&L Attribution ---
    START_PV = "start_pv"                        # float: T-1 portfolio value
    END_PV = "end_pv"                            # float: T portfolio value
    TOTAL_PNL = "total_pnl"                      # float: Total P&L
    DELTA_PNL = "delta_pnl"                      # float: Delta P&L
    GAMMA_PNL = "gamma_pnl"                      # float: Gamma P&L
    VEGA_PNL = "vega_pnl"                        # float: Vega P&L
    THETA_PNL = "theta_pnl"                      # float: Theta P&L
    RHO_PNL = "rho_pnl"                          # float: Rho P&L
    UNEXPLAINED_PNL = "unexplained_pnl"          # float: Unexplained residual
    ATTRIBUTION_REPORT = "attribution_report"    # AttributionReport: Full breakdown
    
    # --- Greeks Validation ---
    ANALYTIC_GREEKS = "analytic_greeks"          # Dict[str, Greeks]: Analytic Greeks
    BUMPED_GREEKS = "bumped_greeks"              # Dict[str, Greeks]: Bump-and-reprice
    VALIDATION_RESULTS = "validation_results"    # Dict[str, Dict]: Comparison results
    DISCREPANCIES = "discrepancies"              # List[Dict]: Positions with issues
    VALIDATION_REPORT = "validation_report"      # ValidationReport: Full report
    
    # =========================================================================
    # CALIBRATION KEYS
    # =========================================================================
    
    # --- Common calibration outputs ---
    CALIBRATION_OBJECTIVE = "calibration_objective"  # CalibrationObjective: Objective fn
    CALIBRATION_RESULT = "calibration_result"    # CalibrationResult: Optimisation result
    CALIBRATED_PARAMS = "calibrated_params"      # Dict[str, float]: Model parameters
    CALIBRATED_SURFACE = "calibrated_surface"    # VolSurface: Calibrated vol surface
    CALIBRATION_ERRORS = "calibration_errors"    # Dict[str, float]: Errors by quote
    
    # --- Heston calibration ---
    HESTON_PARAMS = "heston_params"              # HestonParams: Calibrated Heston params
    MODEL_PRICES = "model_prices"                # Dict[str, float]: Model option prices
    MARKET_PRICES = "market_prices"              # Dict[str, float]: Market option prices
    FELLER_CONDITION = "feller_condition"        # bool: Whether Feller holds
    OPTION_PRICES = "option_prices"              # List[OptionPrice]: Input option data
    
    # --- Hull-White calibration ---
    HULL_WHITE_PARAMS = "hull_white_params"      # HullWhiteParams: Calibrated HW params
    THETA_FUNCTION = "theta_function"            # Callable: Time-dependent θ(t)
    MODEL_SWAPTION_VOLS = "model_swaption_vols"  # Dict: Model swaption vols
    SWAPTION_VOLS = "swaption_vols"              # Dict: Market swaption vols
    
    # =========================================================================
    # MACHINE LEARNING KEYS
    # =========================================================================
    
    # --- Deep Hedging ---
    HEDGING_ENV = "hedging_env"                  # GBMHedgingEnv: Hedging environment
    DEEP_AGENT = "deep_agent"                    # DeepHedgingAgent: Trained agent
    DELTA_AGENT = "delta_agent"                  # DeltaHedgingAgent: Benchmark agent
    TRAINING_RESULT = "training_result"          # Dict: Training history
    EVALUATION_RESULT = "evaluation_result"      # ComparisonResult: Evaluation metrics
    AGENT_PATH = "agent_path"                    # str: Path to saved agent
    COST_MODEL = "cost_model"                    # TransactionCost: Cost model
    RISK_MEASURE = "risk_measure"                # RiskMeasure: Risk measure
    POLICY_NETWORK = "policy_network"            # MLPPolicy: Policy network
    
    # --- GNN Pricer ---
    TRAIN_DATASET = "train_dataset"              # Dataset: Training data
    VAL_DATASET = "val_dataset"                  # Dataset: Validation data
    TEST_DATASET = "test_dataset"                # Dataset: Test data
    MODEL = "model"                              # nn.Module: Trained model
    TRAINING_HISTORY = "training_history"        # Dict: Loss history
    EVALUATION_METRICS = "evaluation_metrics"    # Dict: Test metrics
    MODEL_PATH = "model_path"                    # str: Path to saved model
    
    # --- Calibration Accelerator ---
    CALIBRATION_MODEL = "calibration_model"      # nn.Module: Trained calibration model
    SPEEDUP_FACTOR = "speedup_factor"            # float: Speed improvement
    ACCURACY_METRICS = "accuracy_metrics"        # Dict: Parameter prediction accuracy
    
    # =========================================================================
    # BACKTESTING KEYS
    # =========================================================================
    
    # --- Strategy Backtest ---
    BACKTEST_CONFIG = "backtest_config"          # Dict: Backtest configuration
    BACKTEST_RESULT = "backtest_result"          # BacktestResult: Full result
    PERFORMANCE_METRICS = "performance_metrics"  # PerformanceMetrics: Sharpe, etc.
    TRADE_LOG = "trade_log"                      # DataFrame: All trades executed
    EQUITY_CURVE = "equity_curve"                # Series: Portfolio value over time
    ATTRIBUTION = "attribution"                  # Dict: Return attribution
    STRATEGY = "strategy"                        # Strategy: Trading strategy instance
    
    # --- Model Comparison ---
    MODEL_RESULTS = "model_results"              # Dict[str, PortfolioResult]: By model
    COMPARISON_MATRIX = "comparison_matrix"      # DataFrame: Price/Greek comparison
    CONVERGENCE_ANALYSIS = "convergence_analysis"  # Dict: MC/FDE convergence
    
    # =========================================================================
    # WORKFLOW KEYS
    # =========================================================================
    
    # --- Daily Workflow ---
    DAILY_REPORT = "daily_report"                # Dict: Full daily report
    LIMIT_BREACHES = "limit_breaches"            # List[Dict]: Limit breach alerts
    ALERTS = "alerts"                            # List[str]: Alert messages
    
    # --- Trade Lifecycle ---
    TRADE_REQUEST = "trade_request"              # TradeRequest: New trade request
    TRADE_VALIDATION = "trade_validation"        # Dict: Validation result
    TRADE_BOOKING = "trade_booking"              # Dict: Booking result
    TRADE_LIFECYCLE_REPORT = "trade_lifecycle_report"  # Dict: Full lifecycle report
    
    # --- Hedging Simulation ---
    HEDGING_STRATEGY = "hedging_strategy"        # str: Strategy type
    SIMULATION_PATHS = "simulation_paths"        # np.ndarray: Simulated paths
    HEDGING_PNL = "hedging_pnl"                  # np.ndarray: P&L distribution
    HEDGING_REPORT = "hedging_report"            # Dict: Full hedging report
