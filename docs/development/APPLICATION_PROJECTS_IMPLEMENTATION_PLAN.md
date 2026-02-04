# Application Projects Implementation Plan

**Created:** January 27, 2026  
**Purpose:** Detailed implementation plans for QuantStrata application projects  
**Target:** Professional demonstrations suitable for front-office quant interviews

---

## Overview

This document provides comprehensive implementation plans for the Application Projects defined in `roadmap.md`. Each plan includes:

1. **Methodology** - The quantitative/technical foundation
2. **Workflow** - End-to-end data and process flow
3. **Library Components** - Required QuantStrata modules
4. **UI Structure** - Folder structure within `src/ui/apps/`
5. **UI Features** - Dashboard components and visualisations
6. **Interview Points** - Key talking points for demonstrations

---

## UI Architecture Overview

All application UIs follow the established pattern in `src/ui/`:

```
src/ui/
├── __init__.py              # Package API
├── run.py                   # Entry point: python -m src.ui.run <app_name>
├── _shared/                 # Generic components (existing)
│   ├── layout.py           # Common layout (navbar, footer)
│   ├── styles.py           # CSS / style constants
│   └── components.py       # Reusable components (inputs, cards)
└── apps/                   # Application-specific modules
    ├── pricing_calculator/ # Existing app
    ├── option_analytics/   # Project 1
    ├── algo_trading/       # Project 2
    ├── gnn_lstm_pricer/    # Project 3
    ├── rl_agent/           # Project 4
    ├── vol_trading/        # Project 8
    └── portfolio_opt/      # Project 9
```

---

# Application Project 1: Option Pricing Analytic Report & Visualisation

## Objective

Build a comprehensive option analytics dashboard demonstrating front-office quant capabilities: pricing, Greeks, volatility surfaces, risk metrics, and scenario analysis.

---

## Methodology

### 1. Option Pricing Models

The project demonstrates multiple pricing approaches:

**Black-Scholes-Merton (BSM)**
- Closed-form for European vanilla options
- Formula: `C = S*N(d1) - K*e^(-rT)*N(d2)`
- Used for: Benchmarking, fast Greeks

**Monte Carlo (MC)**
- Path simulation for path-dependent options
- Variance reduction: antithetic, control variates
- Used for: Exotics, barrier, Asian options

**Finite Difference (FD)**
- PDE-based pricing for American options
- PSOR algorithm for early exercise
- Used for: American puts, early exercise premium

### 2. Greeks Calculation

Greeks measure option sensitivities:

| Greek | Definition | Formula/Method |
|-------|-----------|----------------|
| Delta (Δ) | ∂V/∂S | N(d1) for calls |
| Gamma (Γ) | ∂²V/∂S² | n(d1)/(S·σ·√T) |
| Theta (Θ) | ∂V/∂t | Time decay |
| Vega (ν) | ∂V/∂σ | S·√T·n(d1) |
| Rho (ρ) | ∂V/∂r | K·T·e^(-rT)·N(d2) |

**Surface Generation**: Compute Greeks across spot/vol/time grid to create 3D surfaces.

### 3. Volatility Surface

**Implied Volatility**
- Invert BSM price to find σ_implied
- Newton-Raphson iteration

**Surface Calibration**
- SABR: α, β, ρ, ν parameters
- Local Vol: Dupire formula

**Visualisation**
- 3D surface: Strike × Expiry × IV
- Smile by expiry (moneyness vs IV)
- Term structure (ATM vol vs expiry)

### 4. Risk Metrics

**Value at Risk (VaR)**
- Historical: Percentile of historical returns
- Parametric: σ·z_α·√T
- Monte Carlo: Simulated distribution

**Conditional VaR (CVaR)**
- Expected loss beyond VaR
- CVaR = E[Loss | Loss > VaR]

**Scenario Analysis**
- Predefined shocks (spot, vol, rates)
- Custom scenarios
- Fan charts showing outcome distribution

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OPTION ANALYTICS WORKFLOW                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MARKET    │     │   MARKET    │     │  POSITION   │     │   USER      │
│   DATA      │────▶│   SNAPSHOT  │────▶│   SETUP     │────▶│   INPUTS    │
│  Provider   │     │   Dataset   │     │  Portfolio  │     │  Dashboard  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION LAYER                                │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            │
│  │   Curve   │  │    Vol    │  │   SABR    │  │  Dupire   │            │
│  │Bootstrap  │  │  Surface  │  │  Calib    │  │  LocalVol │            │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           PRICING ENGINE                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            │
│  │    BSM    │  │   Monte   │  │   Finite  │  │  Heston   │            │
│  │  Analytic │  │   Carlo   │  │  Diff     │  │  StochVol │            │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           RISK ENGINE                                    │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            │
│  │  Greeks   │  │    VaR    │  │ Scenarios │  │Attribution│            │
│  │ Surfaces  │  │   CVaR    │  │  Stress   │  │   P&L     │            │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        VISUALISATION / OUTPUT                            │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            │
│  │   Dash    │  │   PDF     │  │   HTML    │  │   JSON    │            │
│  │ Dashboard │  │  Report   │  │  Report   │  │    API    │            │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Library Components Required

| Component | Module | Usage |
|-----------|--------|-------|
| Market Data | `src/marketdata/` | Curves, vol surfaces, spot data |
| Instruments | `src/instruments/` | Option definitions |
| Pricers | `src/pricers/` | BSM, MC, FD pricing |
| Calibration | `src/calibration/` | SABR, Dupire, curve bootstrap |
| Risk | `src/risk/` | Greeks, VaR, scenarios |
| Portfolio | `src/portfolio/` | Multi-position management |
| Reporting | `src/core/reporting/` | Plots and visualisations |

---

## UI Structure

```
src/ui/apps/option_analytics/
├── __init__.py              # create_app() factory
├── app.py                   # Main Dash app
├── config.py                # Default parameters
├── tabs/                    # Tab modules
│   ├── __init__.py
│   ├── pricer_tab.py       # Single option pricer
│   ├── greeks_tab.py       # Greeks surfaces
│   ├── vol_surface_tab.py  # Vol surface display/calib
│   ├── scenarios_tab.py    # Scenario analysis
│   └── portfolio_tab.py    # Portfolio risk view
├── callbacks/               # Callback functions
│   ├── __init__.py
│   ├── pricer_callbacks.py
│   ├── greeks_callbacks.py
│   ├── vol_callbacks.py
│   ├── scenario_callbacks.py
│   └── portfolio_callbacks.py
└── utils/                   # Helper functions
    ├── __init__.py
    ├── data_prep.py        # Data transformations
    └── chart_builders.py   # Plotly figure factories
```

---

## UI Features Sketch

### Tab 1: Option Pricer

```
┌─────────────────────────────────────────────────────────────────────────┐
│  OPTION PRICER                                                    [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─ INPUTS ─────────────────┐  ┌─ RESULTS ───────────────────────────┐ │
│  │ Spot:      [100.00    ]  │  │                                     │ │
│  │ Strike:    [100.00    ]  │  │  OPTION PRICE                       │ │
│  │ Expiry:    [0.25      ]  │  │  ┌────────────────────────────────┐ │ │
│  │ Rate (%):  [5.00      ]  │  │  │         $5.3847                │ │ │
│  │ Vol (%):   [20.00     ]  │  │  │         CALL                   │ │ │
│  │                          │  │  └────────────────────────────────┘ │ │
│  │ Type: [Call ▼]           │  │                                     │ │
│  │ Model: [BSM ▼]           │  │  GREEKS                             │ │
│  │                          │  │  ┌──────┬──────┬──────┬──────┐     │ │
│  │ [    CALCULATE    ]      │  │  │  Δ   │  Γ   │  Θ   │  ν   │     │ │
│  │                          │  │  │0.5415│0.0385│-0.012│0.1982│     │ │
│  └──────────────────────────┘  │  └──────┴──────┴──────┴──────┘     │ │
│                                │                                     │ │
│                                └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab 2: Greeks Surface

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GREEKS SURFACE                                                   [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Select Greek: [Delta ▼]  Base Spot: [100]  [GENERATE]                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │                    3D SURFACE PLOT                              │   │
│  │                                                                 │   │
│  │            ▲ Delta                                              │   │
│  │           /│\                                                   │   │
│  │          / │ \                                                  │   │
│  │         /  │  \          ═══════════                           │   │
│  │        /   │   \     ═══             ═══                       │   │
│  │       /    │    ════                     ════                  │   │
│  │      /     │                                                    │   │
│  │     /      │                                                    │   │
│  │    ────────┼────────────────────────────────▶ Spot              │   │
│  │           ╱                                                     │   │
│  │          ╱                                                      │   │
│  │         ▼ Time                                                  │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab 3: Vol Surface

```
┌─────────────────────────────────────────────────────────────────────────┐
│  VOLATILITY SURFACE                                               [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─ 3D Surface ─────────────────────┐ ┌─ Smile by Expiry ────────────┐ │
│  │                                  │ │                              │ │
│  │       IMPLIED VOL SURFACE        │ │    ──── 1M                   │ │
│  │                                  │ │    ──── 3M                   │ │
│  │          ╱═══╲                   │ │    ──── 6M                   │ │
│  │        ╱       ╲                 │ │    ──── 1Y                   │ │
│  │      ╱    ATM    ╲               │ │                              │ │
│  │    ╱───────────────╲             │ │   │\      /│                 │ │
│  │  ─────────────────────           │ │   │ \____/ │                 │ │
│  │                                  │ │   │        │                 │ │
│  │  Strike ──▶      ▲ Expiry        │ │   K/S ───────▶               │ │
│  │                                  │ │                              │ │
│  └──────────────────────────────────┘ └──────────────────────────────┘ │
│                                                                         │
│  ┌─ Term Structure (ATM) ───────────────────────────────────────────┐  │
│  │  Vol │     ╱──────                                               │  │
│  │      │   ╱                                                       │  │
│  │      │ ╱                                                         │  │
│  │      ├──────────────────────────────────────────────────▶ T      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab 4: Scenario Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SCENARIO ANALYSIS                                                [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Position: Call, S=100, K=100, T=0.25, σ=20%     Current Price: $5.38  │
│                                                                         │
│  ┌─ Scenario Results ───────────────────────────────────────────────┐  │
│  │ Scenario           │ Spot   │  Vol  │ Price  │  P&L   │  P&L %  │  │
│  ├────────────────────┼────────┼───────┼────────┼────────┼─────────│  │
│  │ Base Case          │ $100   │  20%  │ $5.38  │  $0.00 │   0.0%  │  │
│  │ Spot +10%          │ $110   │  20%  │ $11.42 │ +$6.04 │ +112.3% │  │
│  │ Spot -10%          │  $90   │  20%  │ $1.58  │ -$3.80 │  -70.6% │  │
│  │ Vol +5%            │ $100   │  25%  │ $6.62  │ +$1.24 │  +23.0% │  │
│  │ Stress: S-20%,V+10%│  $80   │  30%  │ $1.89  │ -$3.49 │  -64.9% │  │
│  └────────────────────┴────────┴───────┴────────┴────────┴─────────┘  │
│                                                                         │
│  ┌─ P&L Chart ──────────────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │  +$6 │        ████                                               │  │
│  │      │        ████                                               │  │
│  │  +$2 │        ████                           ████                │  │
│  │      │        ████                           ████                │  │
│  │   $0 ├────────████───────────────────────────████────────────    │  │
│  │      │                 ████                         ████         │  │
│  │  -$2 │                 ████                         ████         │  │
│  │      │                 ████           ████          ████         │  │
│  │  -$4 │                 ████           ████                       │  │
│  │      └────────────────────────────────────────────────────────   │  │
│  │         Base   S+10%  S-10%  V+5%   V-5%   Stress  Rally         │  │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Interview Points

1. **Pricing Models**: Explain trade-offs between BSM (speed), MC (flexibility), FD (early exercise)
2. **Greeks**: Demonstrate understanding of sensitivities and their practical use in hedging
3. **Vol Surface**: Discuss SABR calibration, smile dynamics, term structure
4. **Risk Metrics**: VaR limitations, CVaR advantages, scenario selection rationale
5. **Architecture**: Clean separation of concerns (data → calibration → pricing → risk → UI)

---

# Application Project 2: Algorithmic Trading Bot

## Objective

Build an algorithmic trading system that demonstrates end-to-end quant workflow: data ingestion, strategy development, backtesting, and simulated execution.

---

## Methodology

### 1. Strategy Development Framework

**Signal Generation**
- Technical indicators (moving averages, RSI, Bollinger)
- Statistical signals (mean reversion, momentum)
- ML-based signals (predictions from trained models)
- RL-based signals (Q-learning agent actions)

**Position Sizing**
- Kelly criterion
- Risk parity
- Volatility targeting

**Execution Logic**
- Market orders vs limit orders
- Order slicing
- TWAP/VWAP execution

### 2. Backtesting Methodology

**Historical Simulation**
- Event-driven architecture
- Realistic order filling (bid-ask spread, slippage)
- Transaction costs

**Performance Metrics**
| Metric | Formula | Target |
|--------|---------|--------|
| Sharpe Ratio | (μ - rf) / σ | > 1.5 |
| Sortino Ratio | (μ - rf) / σ_down | > 2.0 |
| Max Drawdown | max(peak - trough) / peak | < 20% |
| Win Rate | # wins / # trades | > 50% |
| Profit Factor | gross profit / gross loss | > 1.5 |

### 3. ML/RL Integration

**ML Strategies**
- Feature engineering from market data
- Model training (classification/regression)
- Real-time inference

**RL Strategies**
- Environment: market state → action → reward
- Agent: policy network (DQN, PPO, etc.)
- Training: offline on historical data

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ALGORITHMIC TRADING WORKFLOW                        │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │         DATA SOURCES            │
                    │  ┌─────┐  ┌─────┐  ┌─────┐     │
                    │  │Hist │  │Live │  │Alt  │     │
                    │  │Data │  │Feed │  │Data │     │
                    │  └──┬──┘  └──┬──┘  └──┬──┘     │
                    └─────┼───────┼───────┼──────────┘
                          │       │       │
                          ▼       ▼       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMING ENGINE                                 │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │  Data Normalization → Feature Extraction → Signal Queue     │      │
│    └─────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
           ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
           │  TECHNICAL    │ │     ML        │ │     RL        │
           │  STRATEGY     │ │   STRATEGY    │ │   STRATEGY    │
           │               │ │               │ │               │
           │ MA Crossover  │ │ MLPredictor   │ │ RLAgent       │
           │ Momentum      │ │ .predict()    │ │ .act()        │
           │ Mean Revert   │ │               │ │               │
           └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                   │                 │                 │
                   └─────────────────┼─────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PORTFOLIO MANAGER                                 │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │  Signal Aggregation → Position Sizing → Risk Check          │      │
│    └─────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXECUTION ENGINE                                  │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│    │   BACKTEST      │    │     PAPER       │    │      LIVE       │   │
│    │   (Historical)  │    │   (Simulated)   │    │   (Brokerage)   │   │
│    └─────────────────┘    └─────────────────┘    └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PERFORMANCE & REPORTING                             │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │  Equity Curve → Metrics → Trade Log → Risk Report           │      │
│    └─────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Library Components Required

| Component | Module | Usage |
|-----------|--------|-------|
| Backtesting | `src/backtesting/` | Historical simulation |
| Streaming | `src/streaming/` | Live data integration |
| ML | `src/machine_learning/` | ML-based strategies |
| RL | `src/q_learning/` | RL-based strategies |
| Risk | `src/risk/` | Position risk limits |
| Portfolio | `src/portfolio/` | Position management |

---

## UI Structure

```
src/ui/apps/algo_trading/
├── __init__.py
├── app.py
├── config.py
├── tabs/
│   ├── __init__.py
│   ├── strategy_tab.py      # Strategy selection/config
│   ├── backtest_tab.py      # Backtesting controls
│   ├── live_tab.py          # Paper/live trading
│   ├── performance_tab.py   # Results and metrics
│   └── trades_tab.py        # Trade log
├── callbacks/
│   ├── __init__.py
│   ├── strategy_callbacks.py
│   ├── backtest_callbacks.py
│   ├── live_callbacks.py
│   └── performance_callbacks.py
└── utils/
    ├── __init__.py
    ├── strategy_factory.py  # Strategy instantiation
    └── metrics_calc.py      # Performance calculations
```

---

## UI Features Sketch

### Tab 1: Strategy Configuration

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STRATEGY CONFIGURATION                                           [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Strategy Type: [Technical ▼]   Sub-type: [MA Crossover ▼]             │
│                                                                         │
│  ┌─ Parameters ─────────────────┐  ┌─ Preview ───────────────────────┐ │
│  │                              │  │                                 │ │
│  │ Fast MA Period: [10    ]     │  │  Signal preview on sample data  │ │
│  │ Slow MA Period: [50    ]     │  │                                 │ │
│  │ Position Size:  [1000  ]     │  │  ──── Price                     │ │
│  │                              │  │  ━━━━ Fast MA                   │ │
│  │ Risk Limit (%): [2.0   ]     │  │  ╍╍╍╍ Slow MA                   │ │
│  │ Stop Loss (%):  [5.0   ]     │  │  ▲▼   Signals                   │ │
│  │                              │  │                                 │ │
│  │ [  SAVE STRATEGY  ]          │  │  [chart preview here]           │ │
│  │                              │  │                                 │ │
│  └──────────────────────────────┘  └─────────────────────────────────┘ │
│                                                                         │
│  ── OR ── Select ML/RL Strategy:                                       │
│                                                                         │
│  [x] Use ML Predictor    Model: [trained_model_v1.h5 ▼]               │
│  [ ] Use RL Agent        Agent: [dqn_agent_v2 ▼]                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab 2: Backtesting

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BACKTESTING                                                      [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Symbol: [AAPL ▼]  Start: [2023-01-01]  End: [2024-01-01]              │
│  Initial Capital: [$100,000]   Commission: [0.001]                      │
│                                                                         │
│  [       RUN BACKTEST       ]                                          │
│                                                                         │
│  ┌─ Equity Curve ───────────────────────────────────────────────────┐  │
│  │                                                        ╱──       │  │
│  │  $120K │                                          ╱───╱          │  │
│  │        │                              ╱──────────╱               │  │
│  │  $100K ├─────────╱───────────────────╱                           │  │
│  │        │        ╱                                                │  │
│  │   $80K │       ╱                                                 │  │
│  │        └────────────────────────────────────────────────────▶    │  │
│  │          Jan   Mar   May   Jul   Sep   Nov   Jan                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Metrics ────────────┐  ┌─ Drawdown ─────────────────────────────┐  │
│  │ Total Return: +18.5% │  │                                        │  │
│  │ Sharpe:       1.82   │  │    0% ├────────────────────────────    │  │
│  │ Sortino:      2.41   │  │   -5% │    ╲  ╱╲      ╱╲               │  │
│  │ Max DD:      -8.3%   │  │  -10% │     ╲╱  ╲────╱                 │  │
│  │ Win Rate:     58%    │  │       └───────────────────────────▶    │  │
│  │ # Trades:     127    │  │                                        │  │
│  └──────────────────────┘  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Interview Points

1. **Event-Driven Architecture**: Explain how the system processes market events
2. **Strategy Abstraction**: Discuss the strategy interface and how different types plug in
3. **Risk Management**: Describe position limits, stop losses, and portfolio constraints
4. **ML/RL Integration**: Show how trained models are deployed in live trading
5. **Performance Attribution**: Explain how to decompose returns by factor/strategy

---

# Application Project 3: Hybrid GNN-LSTM Full Revaluation Pricer

## Objective

Demonstrate a cutting-edge ML-based pricer that uses graph neural networks (GNN) and LSTM to value entire portfolios, capturing trade interdependencies.

---

## Methodology

### 1. Portfolio Graph Representation

**Node Features** (per trade):
- Trade type (call, put, forward, etc.)
- Moneyness (S/K or log-moneyness)
- Time to expiry
- Notional
- Greeks (delta, gamma, vega)

**Edge Features** (trade relationships):
- Same underlying → edge weight based on correlation
- Offsetting positions → negative weight
- Same expiry bucket → temporal edge

### 2. Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GNN-LSTM HYBRID ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────┘

Input Layer
┌─────────────────────────────────────────────────────────────────────────┐
│  Trade Features (N trades × F features)                                  │
│  Market Data Time Series (T timesteps × M market features)              │
│  Adjacency Matrix (N × N)                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
GNN Layers (Graph Convolution)
┌─────────────────────────────────────────────────────────────────────────┐
│  GraphConv → ReLU → GraphConv → ReLU                                    │
│  Output: Trade embeddings capturing portfolio structure                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
LSTM Layers (Temporal Processing)
┌─────────────────────────────────────────────────────────────────────────┐
│  LSTM(64) → LSTM(32)                                                    │
│  Output: Temporal market state embedding                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Fusion Layer
┌─────────────────────────────────────────────────────────────────────────┐
│  Concatenate [Trade Embeddings, Market Embedding]                       │
│  Attention mechanism to weight trade importance                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Output Layer
┌─────────────────────────────────────────────────────────────────────────┐
│  Dense → Portfolio Value                                                │
│  (Or per-trade values that sum to portfolio)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. Training Methodology

**Data Generation**
1. Generate synthetic portfolios (varying sizes, compositions)
2. Price each trade with library pricers (ground truth)
3. Sum for portfolio value

**Training Process**
1. Split: 70% train, 15% validation, 15% test
2. Loss: MSE between predicted and library portfolio value
3. Regularization: Dropout, L2, early stopping

**Validation**
- Compare vs library pricers on test set
- Measure per-trade error distribution
- Speed comparison (GNN vs full revaluation)

---

## Library Components Required

| Component | Module | Usage |
|-----------|--------|-------|
| GNN-LSTM Model | `src/machine_learning/models/gnn_rnn_hybrid/` | Core model |
| Trade Graph Builder | `src/machine_learning/utilities/trade_graph_builder.py` | Graph construction |
| Trade Encoder | `src/machine_learning/utilities/trade_attribute_encoder.py` | Feature encoding |
| Portfolio | `src/portfolio/` | Portfolio representation |
| Pricers | `src/pricers/` | Ground truth pricing |
| Training | `src/machine_learning/training/` | Training pipeline |

---

## UI Structure

```
src/ui/apps/gnn_lstm_pricer/
├── __init__.py
├── app.py
├── config.py
├── tabs/
│   ├── __init__.py
│   ├── portfolio_tab.py     # Portfolio input/generation
│   ├── graph_tab.py         # Graph visualisation
│   ├── inference_tab.py     # Run model inference
│   ├── comparison_tab.py    # GNN vs library comparison
│   └── training_tab.py      # Training monitoring
├── callbacks/
│   ├── __init__.py
│   ├── portfolio_callbacks.py
│   ├── graph_callbacks.py
│   ├── inference_callbacks.py
│   └── comparison_callbacks.py
└── utils/
    ├── __init__.py
    ├── graph_viz.py         # Cytoscape/NetworkX visualisation
    └── comparison_calc.py   # Error metrics
```

---

## UI Features Sketch

### Tab: Graph Visualisation

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO GRAPH                                                  [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Portfolio: [Sample Portfolio ▼]   # Trades: 15                        │
│                                                                         │
│  ┌─ Trade Graph ────────────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │              ○ Call_AAPL_100                                     │  │
│  │             ╱│╲                                                   │  │
│  │            ╱ │ ╲                                                  │  │
│  │           ╱  │  ╲                                                 │  │
│  │          ○───┼───○ Put_AAPL_95                                   │  │
│  │   Call_MSFT  │                                                    │  │
│  │          ╲   │   ╱                                                │  │
│  │           ╲  │  ╱                                                 │  │
│  │            ╲ │ ╱                                                  │  │
│  │             ╲│╱                                                   │  │
│  │              ○ Fwd_SPX                                           │  │
│  │                                                                   │  │
│  │  Legend: ○ Trade Node   ─── Same Underlying   ═══ Same Expiry    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Node Details ───────────────┐  ┌─ Edge Summary ─────────────────┐  │
│  │ Selected: Call_AAPL_100      │  │ # Edges: 42                    │  │
│  │ Moneyness: 1.02              │  │ Avg Weight: 0.35               │  │
│  │ TTM: 0.25                    │  │ Max Weight: 0.92               │  │
│  │ Delta: 0.54                  │  │ Clusters: 3                    │  │
│  └──────────────────────────────┘  └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab: Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GNN vs LIBRARY COMPARISON                                        [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [   RUN COMPARISON   ]        Progress: ████████░░ 80%                │
│                                                                         │
│  ┌─ Summary ────────────────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │  Portfolio Value (Library): $1,234,567.89                        │  │
│  │  Portfolio Value (GNN):     $1,231,042.15                        │  │
│  │                                                                   │  │
│  │  Absolute Error:  $3,525.74  (0.29%)                             │  │
│  │  Inference Time:  0.015s vs 2.34s (156x faster)                  │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Per-Trade Error Distribution ───────────────────────────────────┐  │
│  │                                                                   │  │
│  │  Count │                                                         │  │
│  │        │      ████                                               │  │
│  │        │    ████████                                             │  │
│  │        │  ████████████                                           │  │
│  │        │ ██████████████                                          │  │
│  │        └───────────────────────────────────────────────▶         │  │
│  │           -2%  -1%   0%   +1%  +2%                               │  │
│  │                  Error (%)                                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Interview Points

1. **Why GNN?**: Graph structure captures portfolio interdependencies (hedges, correlations)
2. **Why LSTM?**: Captures temporal dynamics of market data
3. **Fusion Strategy**: Explain attention-based fusion of trade and market embeddings
4. **Speed vs Accuracy Trade-off**: When is GNN appropriate vs traditional pricing?
5. **Production Deployment**: Model serving, monitoring, retraining strategy

---

# Application Project 4: Q-Learning Orchestrator Agent

## Objective

Demonstrate reinforcement learning for quantitative finance applications: delta hedging and algorithmic trading with trained RL agents.

---

## Methodology

### 1. RL Framework

**Markov Decision Process (MDP)**
- State (s): Market observation + position
- Action (a): Trade decision (buy/sell/hold or continuous sizing)
- Reward (r): Risk-adjusted P&L
- Transition (s'): Next market state

**Algorithms**
- DQN: Discrete action space, experience replay
- PPO: Continuous actions, policy gradient
- Custom: Domain-specific modifications

### 2. Delta Hedging Application

**Environment**: `HedgingEnvironment`
- State: spot, time, position, delta, gamma, vol
- Action: hedge ratio (continuous)
- Reward: -|PnL variance| - transaction_cost

**Objective**: Learn hedging policy that minimises variance of terminal P&L while accounting for transaction costs.

**Benchmark**: Delta-neutral hedging (always hedge to delta=0)

### 3. Trading Application

**Environment**: `TradingEnvironment`
- State: price history, indicators, position
- Action: position change
- Reward: Sharpe ratio or risk-adjusted return

**Objective**: Learn trading policy that maximises risk-adjusted returns.

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RL AGENT WORKFLOW                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                                   │
└─────────────────────────────────────────────────────────────────────────┘

         ┌───────────────────────────────────────────────┐
         │              ENVIRONMENT                       │
         │  ┌─────────────────────────────────────────┐  │
         │  │ State: [spot, time, pos, delta, ...]    │  │
         │  └─────────────────────┬───────────────────┘  │
         └────────────────────────┼──────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             │             │
         ┌──────────────────┐    │    ┌────────┴────────┐
         │      AGENT       │    │    │     REWARD      │
         │  ┌────────────┐  │    │    │   r = f(s, a)   │
         │  │   Policy   │  │    │    └────────┬────────┘
         │  │   π(s)     │──┼────┘             │
         │  └────────────┘  │                  │
         │        │         │                  │
         │        ▼         │                  │
         │  ┌────────────┐  │                  │
         │  │   Action   │  │◄─────────────────┘
         │  │    a       │──┼──────────────────┐
         │  └────────────┘  │                  │
         └──────────────────┘                  │
                                               ▼
                                  ┌─────────────────────┐
                                  │   EXPERIENCE REPLAY  │
                                  │  (s, a, r, s')      │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │    POLICY UPDATE    │
                                  │   (Gradient Step)   │
                                  └─────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT PHASE                                  │
└─────────────────────────────────────────────────────────────────────────┘

  Market Data ──▶ Environment ──▶ Agent.act(state) ──▶ Execution
       │                                                    │
       └────────────────────────────────────────────────────┘
                           (feedback loop)
```

---

## Library Components Required

| Component | Module | Usage |
|-----------|--------|-------|
| RL Framework | `src/q_learning/` | Core RL infrastructure |
| Environments | `src/q_learning/environments/` | Trading, Hedging envs |
| Runners | `src/q_learning/runners/` | Backtest, Live execution |
| Deep Hedging | `src/deep_hedging/` | Hedging-specific components |
| Pricers | `src/pricers/` | Option pricing for hedging |
| Backtesting | `src/backtesting/` | Historical evaluation |

---

## UI Structure

```
src/ui/apps/rl_agent/
├── __init__.py
├── app.py
├── config.py
├── tabs/
│   ├── __init__.py
│   ├── environment_tab.py   # Environment configuration
│   ├── training_tab.py      # Training controls & monitoring
│   ├── evaluation_tab.py    # Agent evaluation & benchmarks
│   ├── replay_tab.py        # Episode replay visualisation
│   └── deployment_tab.py    # Live/paper deployment
├── callbacks/
│   ├── __init__.py
│   ├── environment_callbacks.py
│   ├── training_callbacks.py
│   ├── evaluation_callbacks.py
│   └── replay_callbacks.py
└── utils/
    ├── __init__.py
    ├── training_viz.py      # Training curve plots
    └── episode_viz.py       # Episode playback
```

---

## UI Features Sketch

### Tab: Training Monitor

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRAINING MONITOR                                                 [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Agent: [DeepHedging_DQN ▼]   Environment: [GBMHedging ▼]              │
│                                                                         │
│  [  START TRAINING  ]  [  STOP  ]      Episodes: 1,247 / 5,000         │
│                                                                         │
│  ┌─ Reward Curve ───────────────────────────────────────────────────┐  │
│  │                                                         ╱────    │  │
│  │  Reward │                                          ╱───╱         │  │
│  │         │                             ╱───────────╱              │  │
│  │         │           ╱────────────────╱                           │  │
│  │         │     ╱────╱                                             │  │
│  │         │ ───╱                                                   │  │
│  │         └────────────────────────────────────────────────▶       │  │
│  │            0       1000     2000     3000     4000    Episodes   │  │
│  │                                                                   │  │
│  │  ──── Mean Reward    ╍╍╍╍ Moving Avg (100)                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Training Stats ─────────┐  ┌─ Hyperparameters ──────────────────┐  │
│  │ Best Reward:     -0.023  │  │ Learning Rate:  0.0003             │  │
│  │ Current ε:       0.15    │  │ Discount (γ):   0.99               │  │
│  │ Loss:            0.0042  │  │ Batch Size:     64                 │  │
│  │ Steps/sec:       1,250   │  │ Buffer Size:    100,000            │  │
│  └──────────────────────────┘  └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab: Episode Replay

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EPISODE REPLAY                                                   [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Episode: [1247 ▼]    [  ◀  ] [  ▶  ] [  ▶▶  ]    Step: 35 / 50       │
│                                                                         │
│  ┌─ Price & Position ───────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │  $105 │              ╱╲                                          │  │
│  │       │            ╱╱  ╲╲         ╱╱                             │  │
│  │  $100 │     ╱╲   ╱╱      ╲╲     ╱╱  ╲╲                           │  │
│  │       │   ╱╱  ╲╲╱╱         ╲╲  ╱╱     ╲╲                         │  │
│  │   $95 │ ╱╱                   ╲╱                                  │  │
│  │       └───────────────────────────────────────────────────▶      │  │
│  │                                                                   │  │
│  │  ──── Spot Price    ▲▼ Agent Actions                             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ State @ Step 35 ────────┐  ┌─ Episode Summary ──────────────────┐  │
│  │ Spot:      $102.34       │  │ Terminal PnL:    -$234.56          │  │
│  │ Position:  0.52 delta    │  │ Total Cost:      $89.12            │  │
│  │ Delta:     0.58          │  │ # Trades:        23                │  │
│  │ Time:      0.15          │  │ vs Benchmark:    +$45.23 better    │  │
│  │ Action:    Buy 0.06      │  │                                    │  │
│  └──────────────────────────┘  └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab: Benchmark Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BENCHMARK COMPARISON                                             [Tab] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Evaluate on: [1000] episodes   [  RUN EVALUATION  ]                   │
│                                                                         │
│  ┌─ P&L Distribution ───────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │  Count │        ░░░░░░░░░                                        │  │
│  │        │      ░░░░░░░░░░░░░░                                     │  │
│  │        │    ░░░░░░░░░░░░░░░░░░  ████████                         │  │
│  │        │  ░░░░░░░░░░░░░░░░░░░░  ██████████████                   │  │
│  │        │ ░░░░░░░░░░░░░░░░░░░░░░ ████████████████████             │  │
│  │        └────────────────────────────────────────────────▶        │  │
│  │                          P&L ($)                                  │  │
│  │                                                                   │  │
│  │  ░░░░ Delta Hedge (Benchmark)    ████ RL Agent                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─ Metrics Comparison ─────────────────────────────────────────────┐  │
│  │                      │  Delta Hedge  │   RL Agent   │  Δ Improv  │  │
│  │ Mean PnL             │     -$45.23   │    -$23.45   │   +48.1%   │  │
│  │ Std PnL              │     $234.56   │    $198.34   │   +15.4%   │  │
│  │ Sharpe               │       -0.19   │      -0.12   │   +36.8%   │  │
│  │ CVaR (95%)           │    -$523.45   │   -$412.67   │   +21.2%   │  │
│  │ Total Cost           │     $123.45   │     $89.12   │   +27.8%   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Interview Points

1. **MDP Formulation**: Explain state/action/reward design for hedging/trading
2. **Exploration vs Exploitation**: ε-greedy, entropy bonus, etc.
3. **Reward Engineering**: Why variance-based reward for hedging?
4. **Sample Efficiency**: Experience replay, prioritised sampling
5. **Deployment Challenges**: Distribution shift, risk limits in production

---

# Application Project 8: Volatility Trading & Variance Swap Analytics

## Objective

Provide volatility trading analytics: variance swap pricing, dispersion analysis, and vol-of-vol metrics.

---

## Methodology

### 1. Variance Swap Pricing

**Replicating Portfolio Method** (Carr-Madan):
```
Fair Variance = (2/T) × [∫₀^F (1/K²)P(K)dK + ∫_F^∞ (1/K²)C(K)dK]
```

Where P(K), C(K) are OTM put/call prices.

**Discrete Adjustment**:
```
σ²_discrete ≈ σ²_continuous × (1 + 1/(3N))
```

### 2. Dispersion Trading

**Implied Correlation**:
```
σ²_index = Σᵢ Σⱼ wᵢwⱼρᵢⱼσᵢσⱼ
```

Solve for implied ρ given index vol and constituent vols.

**Strategy**:
- Sell index variance (expect realised < implied)
- Buy constituent variance (hedge vega exposure)
- Profit if implied correlation > realised correlation

### 3. Vol-of-Vol Analysis

**VVIX-style metrics**:
- Vol of implied vol
- Vol clustering / persistence
- Regime detection (low/normal/high/crisis)

---

## Library Components Required

| Component | Module | Usage |
|-----------|--------|-------|
| Variance Swap | `src/volatility/trading/variance_swap.py` | Pricing |
| Dispersion | `src/volatility/trading/dispersion.py` | Analysis |
| Vol-of-Vol | `src/volatility/analytics/vol_of_vol.py` | Metrics |
| Vol Surface | `src/marketdata/surfaces/` | Market data |
| Calibration | `src/calibration/` | SABR, Heston |

---

## UI Structure

```
src/ui/apps/vol_trading/
├── __init__.py
├── app.py
├── config.py
├── tabs/
│   ├── __init__.py
│   ├── variance_swap_tab.py    # Var swap pricing
│   ├── dispersion_tab.py       # Dispersion analysis
│   ├── vol_of_vol_tab.py       # VoV metrics
│   └── surface_tab.py          # Vol surface analytics
├── callbacks/
│   └── ...
└── utils/
    └── ...
```

---

# Application Project 9: Portfolio Construction & Optimisation

## Objective

Demonstrate portfolio optimisation capabilities: mean-variance, risk parity, and Black-Litterman.

---

## Methodology

### 1. Mean-Variance Optimisation

**Markowitz Problem**:
```
min  w'Σw           (variance)
s.t. w'μ ≥ r_target (return constraint)
     w'1 = 1        (budget)
     w ≥ 0          (long-only)
```

**Efficient Frontier**: Vary r_target to trace frontier.

### 2. Risk Parity

**Equal Risk Contribution**:
```
RC_i = w_i × (Σw)_i / σ_p = 1/N   for all i
```

Solve iteratively to find weights where each asset contributes equally to portfolio risk.

### 3. Black-Litterman

**Prior**: Market equilibrium returns
```
π = δ × Σ × w_mkt
```

**Posterior** (incorporating views):
```
μ_post = π + τΣP'(PτΣP' + Ω)⁻¹(Q - Pπ)
```

---

## Library Components Required

| Component | Module | Usage |
|-----------|--------|-------|
| Mean-Variance | `src/portfolio/optimization/mean_variance.py` | MV optimisation |
| Risk Parity | `src/portfolio/optimization/risk_parity.py` | RP optimisation |
| Black-Litterman | `src/portfolio/optimization/black_litterman.py` | BL model |
| Covariance | `src/portfolio/optimization/covariance.py` | Cov estimation |

---

## UI Structure

```
src/ui/apps/portfolio_opt/
├── __init__.py
├── app.py
├── config.py
├── tabs/
│   ├── __init__.py
│   ├── mean_variance_tab.py    # MV optimisation
│   ├── risk_parity_tab.py      # Risk parity
│   ├── black_litterman_tab.py  # BL with views
│   ├── efficient_frontier_tab.py # Frontier visualisation
│   └── backtest_tab.py         # Portfolio backtest
├── callbacks/
│   └── ...
└── utils/
    └── ...
```

---

# Summary: Implementation Roadmap

## Priority Order

1. **Option Analytics Dashboard** (Project 1) - Core demonstration of library
2. **RL Agent Dashboard** (Project 4) - Cutting-edge ML/RL showcase
3. **Vol Trading Dashboard** (Project 8) - Uses new Phase 8.1 components
4. **Portfolio Optimisation Dashboard** (Project 9) - Uses new Phase 8.2 components
5. **Algo Trading Dashboard** (Project 2) - Complex integration
6. **GNN-LSTM Pricer Dashboard** (Project 3) - Advanced ML showcase

## Estimated Effort per Project

| Project | Tabs | Callbacks | Utils | Total Estimate |
|---------|------|-----------|-------|----------------|
| 1. Option Analytics | 5 | 5 | 2 | 8-10 hours |
| 2. Algo Trading | 5 | 4 | 2 | 10-12 hours |
| 3. GNN-LSTM Pricer | 5 | 4 | 2 | 8-10 hours |
| 4. RL Agent | 5 | 4 | 2 | 8-10 hours |
| 8. Vol Trading | 4 | 4 | 1 | 6-8 hours |
| 9. Portfolio Opt | 5 | 4 | 1 | 6-8 hours |

---

## Next Steps

1. **Fill Implementation Checklist Gaps** (tests, docs, pipelines)
2. **Create `src/ui/apps/<project>/` structure** for each project
3. **Implement core app.py** with layout and basic callbacks
4. **Add tab modules** with specific functionality
5. **Polish visualisations** for interview demonstrations
6. **Create example notebooks** showing workflows

---

*This plan should be reviewed and updated as implementation progresses.*
