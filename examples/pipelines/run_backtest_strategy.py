#!/usr/bin/env python3
"""
===============================================================================
Pipeline Example: backtest.run_strategy
===============================================================================

This script demonstrates how to use the `backtest.run_strategy` pipeline
to backtest a trading strategy over historical data.

What This Pipeline Does
-----------------------
1. Loads historical market data
2. Initializes the trading strategy
3. Runs the backtest simulation day-by-day
4. Records trades, positions, and P&L
5. Calculates performance metrics
6. Writes comprehensive backtest report

Backtest Components
-------------------
- **Strategy**: Trading logic that generates signals
- **Portfolio**: Position tracking and constraints
- **Market Data**: Historical prices, vols, rates
- **Execution Model**: How trades are filled (slippage, costs)
- **Risk Model**: Position sizing and limits

Performance Metrics
-------------------
- **Returns**: Total return, CAGR, daily/monthly returns
- **Risk**: Volatility, max drawdown, VaR
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trading**: Win rate, profit factor, avg trade

Why Backtest?
-------------
- Validate strategy hypothesis before live trading
- Estimate expected returns and risk
- Optimize parameters (carefully, avoid overfitting!)
- Build confidence and understand edge

Run This Example
----------------
    python examples/pipelines/run_backtest_strategy.py

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.core.state_keys import StateKeys as Keys


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config() -> RunConfig:
    """
    Build configuration for strategy backtest.
    
    We'll backtest a simple options selling strategy:
    - Sell short-dated ATM straddles
    - Delta hedge daily
    - Close at expiry and roll
    """
    
    # -------------------------------------------------------------------------
    # Define the trading strategy
    # -------------------------------------------------------------------------
    strategy_config = {
        # Strategy identifier
        "name": "short_straddle_delta_hedged",
        
        # Strategy type (determines which strategy class to use)
        "type": "options_selling",
        
        # Strategy parameters
        "params": {
            # What to sell
            "position_type": "straddle",          # "straddle", "strangle", "call", "put"
            "option_style": "european",
            
            # Strike selection
            "strike_method": "atm",               # "atm", "delta", "fixed"
            "delta_target": None,                 # Used if strike_method == "delta"
            
            # Expiry selection
            "target_dte": 30,                     # Target days to expiry
            "min_dte": 7,                         # Close when DTE falls below
            "max_dte": 45,                        # Don't open if DTE exceeds
            
            # Position sizing
            "sizing_method": "notional",          # "notional", "premium", "vega"
            "target_notional": 1_000_000,         # $1M per position
            
            # Hedging
            "delta_hedge": True,                  # Hedge delta?
            "hedge_frequency": "daily",           # "daily", "threshold", "none"
            "delta_threshold": 0.05,              # Rehedge if delta moves more than 5%
            "gamma_hedge": False,                 # Hedge gamma? (expensive)
        },
    }
    
    # -------------------------------------------------------------------------
    # Market data configuration
    # -------------------------------------------------------------------------
    market_data_config = {
        # Data source
        "source": "synthetic",                    # "synthetic", "csv", "database"
        
        # Asset to trade
        "underlying": "EURUSD",
        
        # Simulation period
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        
        # Synthetic data parameters (if source == "synthetic")
        "synthetic": {
            "initial_spot": 1.10,
            "drift": 0.02,                        # Annual drift
            "volatility": 0.10,                   # Annual vol
            "vol_of_vol": 0.30,                   # Vol regime changes
        },
    }
    
    # -------------------------------------------------------------------------
    # Execution model
    # -------------------------------------------------------------------------
    execution_config = {
        # Bid-ask spread (half-spread)
        "spread_bps": 2.0,                        # 0.02%
        
        # Slippage model
        "slippage_model": "proportional",         # "fixed", "proportional", "market_impact"
        "slippage_bps": 1.0,
        
        # Transaction costs
        "commission_per_trade": 0.0,              # Fixed commission
        "commission_bps": 0.5,                    # Proportional commission
    }
    
    # -------------------------------------------------------------------------
    # Build the RunConfig
    # -------------------------------------------------------------------------
    config = RunConfig(
        pipeline="backtest.run_strategy",
        
        io={
            "artifacts_dir": "./artifacts/backtest_example",
            "enable_save": True,
        },
        
        params={
            "backtest": {
                # The strategy to run
                "strategy": strategy_config,
                
                # Market data settings
                "market_data": market_data_config,
                
                # Trade execution
                "execution": execution_config,
                
                # Initial capital
                "initial_capital": 10_000_000,    # $10M
                
                # Risk limits
                "risk_limits": {
                    "max_position_size": 0.20,    # Max 20% of capital per position
                    "max_portfolio_delta": 0.30,  # Max 30% portfolio delta
                    "stop_loss": 0.10,            # Stop if down 10% on trade
                    "max_drawdown": 0.20,         # Halt if down 20% portfolio
                },
                
                # Reporting
                "benchmark": "cash",              # Compare against: "cash", "buy_hold", "index"
                "report_frequency": "daily",
            }
        },
    )
    
    return validate_run_config(config)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Execute the backtest pipeline and display results."""
    
    print("=" * 70)
    print("Pipeline Example: Strategy Backtest")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Build configuration
    # -------------------------------------------------------------------------
    print("[1/4] Building configuration...")
    cfg = build_config()
    print(f"      Pipeline: {cfg.pipeline}")
    print(f"      Strategy: Short Straddle (Delta Hedged)")
    print(f"      Period: 2020-01-01 to 2023-12-31 (4 years)")
    print(f"      Initial capital: $10,000,000")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Execute the pipeline
    # -------------------------------------------------------------------------
    print("[2/4] Running backtest...")
    print("      (Simulating ~1000 trading days)")
    ctx = run_pipeline_from_config(cfg)
    print("      Backtest completed!")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Extract results
    # -------------------------------------------------------------------------
    print("[3/4] Extracting results...")
    
    backtest_result = ctx.state.get(Keys.BACKTEST_RESULT, {})
    equity_curve = ctx.state.get(Keys.EQUITY_CURVE, [])
    trade_log = ctx.state.get(Keys.TRADE_LOG, [])
    performance_metrics = ctx.state.get(Keys.PERFORMANCE_METRICS, {})
    
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Display Backtest Results
    # -------------------------------------------------------------------------
    print("[4/4] Backtest Results Report")
    print("=" * 70)
    print()
    
    # Performance summary
    print("Performance Summary:")
    print("-" * 60)
    
    total_return = performance_metrics.get("total_return", 0)
    cagr = performance_metrics.get("cagr", 0)
    volatility = performance_metrics.get("volatility", 0)
    sharpe = performance_metrics.get("sharpe_ratio", 0)
    sortino = performance_metrics.get("sortino_ratio", 0)
    max_drawdown = performance_metrics.get("max_drawdown", 0)
    calmar = performance_metrics.get("calmar_ratio", 0)
    
    initial_capital = 10_000_000
    final_capital = initial_capital * (1 + total_return)
    
    print(f"  Initial Capital:    ${initial_capital:>15,.0f}")
    print(f"  Final Capital:      ${final_capital:>15,.0f}")
    print(f"  Total Return:       {total_return:>15.2%}")
    print(f"  CAGR:               {cagr:>15.2%}")
    print()
    
    # Risk metrics
    print("Risk Metrics:")
    print("-" * 60)
    print(f"  Annualized Vol:     {volatility:>15.2%}")
    print(f"  Max Drawdown:       {max_drawdown:>15.2%}")
    print(f"  Sharpe Ratio:       {sharpe:>15.2f}")
    print(f"  Sortino Ratio:      {sortino:>15.2f}")
    print(f"  Calmar Ratio:       {calmar:>15.2f}")
    print()
    
    # Trading statistics
    print("Trading Statistics:")
    print("-" * 60)
    
    n_trades = performance_metrics.get("n_trades", 0)
    win_rate = performance_metrics.get("win_rate", 0)
    avg_win = performance_metrics.get("avg_win", 0)
    avg_loss = performance_metrics.get("avg_loss", 0)
    profit_factor = performance_metrics.get("profit_factor", 0)
    avg_holding_days = performance_metrics.get("avg_holding_period", 0)
    
    print(f"  Total Trades:       {n_trades:>15,d}")
    print(f"  Win Rate:           {win_rate:>15.1%}")
    print(f"  Average Win:        ${avg_win:>14,.0f}")
    print(f"  Average Loss:       ${avg_loss:>14,.0f}")
    print(f"  Profit Factor:      {profit_factor:>15.2f}")
    print(f"  Avg Holding Period: {avg_holding_days:>12.1f} days")
    print()
    
    # Monthly returns (sample)
    print("Monthly Returns (Sample):")
    print("-" * 60)
    
    monthly_returns = performance_metrics.get("monthly_returns", {})
    if monthly_returns:
        # Show last 12 months
        months = list(monthly_returns.items())[-12:]
        print(f"  {'Month':<12} {'Return':>12}")
        print(f"  {'-'*12:<12} {'-'*12:>12}")
        for month, ret in months:
            color = "+" if ret > 0 else " "
            print(f"  {month:<12} {color}{ret:>11.2%}")
    print()
    
    # -------------------------------------------------------------------------
    # Strategy interpretation
    # -------------------------------------------------------------------------
    print("Strategy Analysis:")
    print("-" * 70)
    
    # Sharpe assessment
    if sharpe > 2.0:
        sharpe_comment = "EXCELLENT - Exceptional risk-adjusted returns"
    elif sharpe > 1.0:
        sharpe_comment = "GOOD - Strong risk-adjusted returns"
    elif sharpe > 0.5:
        sharpe_comment = "MODERATE - Acceptable for some investors"
    elif sharpe > 0:
        sharpe_comment = "WEAK - May not justify the risk"
    else:
        sharpe_comment = "POOR - Negative risk-adjusted returns"
    
    print(f"  Sharpe assessment: {sharpe_comment}")
    
    # Win rate + profit factor
    if win_rate > 0.5 and profit_factor > 1.5:
        trade_comment = "High win rate with good profit factor"
    elif profit_factor > 2.0:
        trade_comment = "Let winners run, cut losers (trend-following profile)"
    else:
        trade_comment = "Monitor closely - edge may be marginal"
    
    print(f"  Trade profile: {trade_comment}")
    
    # Drawdown assessment
    if max_drawdown < 0.10:
        dd_comment = "LOW RISK - Very controlled drawdowns"
    elif max_drawdown < 0.20:
        dd_comment = "MODERATE RISK - Within acceptable limits"
    else:
        dd_comment = "HIGH RISK - May be uncomfortable for investors"
    
    print(f"  Drawdown profile: {dd_comment}")
    
    print()
    print("Artifacts saved to:", cfg.io.get("artifacts_dir", "N/A"))
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
