# QuantStrata Ecosystem Architecture

This document provides visual diagrams showing how all library components interact. Each section covers a different aspect of the architecture.

---

## 1. High-Level Module Overview

The library is organised into **core infrastructure** (data, instruments, models), **computation engines** (pricers, risk, calibration), and **execution frameworks** (portfolio, backtesting, streaming, orchestrator, ML).

```mermaid
graph TB
    subgraph foundation [Foundation Layer]
        MD[marketdata]
        INST[instruments]
        MOD[models]
    end

    subgraph computation [Computation Layer]
        PRC[pricers]
        CAL[calibration]
    end

    subgraph execution [Execution Layer]
        PORT[portfolio]
        RISK[risk]
        BT[backtesting]
        STR[streaming]
        ML[machine_learning]
    end

    subgraph orchestration [Orchestration Layer]
        ORCH[orchestrator]
    end

    subgraph interface [Interface Layer]
        UI[ui]
        CORE[core/reporting]
    end

    MD --> PRC
    INST --> PRC
    MOD --> PRC

    MD --> CAL
    MOD --> CAL

    PRC --> PORT
    PORT --> RISK
    MD --> RISK

    MD --> BT
    PORT --> BT

    MD --> STR
    PORT --> STR

    PORT --> ML
    PRC --> ML
    MD --> ML

    PORT --> ORCH
    RISK --> ORCH
    MD --> ORCH
    BT --> ORCH

    PORT --> UI
    MD --> UI
    PRC --> UI
    RISK --> CORE
```

---

## 2. Module Dependency Graph

Shows what each module imports from other modules.

```mermaid
graph LR
    subgraph no_deps [No External Dependencies]
        MD_CORE[marketdata.core]
        INST_CORE[instruments.core]
        MOD_COMMON[models.common]
    end

    subgraph level1 [Level 1: Core Data]
        MD_CURVES[marketdata.curves]
        MD_SURFACES[marketdata.surfaces]
        MD_PROVIDERS[marketdata.providers]
        MD_SCENARIOS[marketdata.scenarios]
    end

    subgraph level2 [Level 2: Instruments and Models]
        INST_FX[instruments.fx]
        INST_EQ[instruments.equity]
        INST_IR[instruments.ir]
        MOD_ANALYTIC[models.analytic]
        MOD_STOCHVOL[models.stochastic_volatility]
        MOD_NUMERIC[models.numeric]
        MOD_PAYOFFS[models.payoffs]
    end

    subgraph level3 [Level 3: Pricers]
        PRC_FX[pricers.fx]
        PRC_EQ[pricers.equity]
        PRC_IR[pricers.ir]
        PRC_REG[pricers.registry]
    end

    subgraph level4 [Level 4: Portfolio and Calibration]
        PORT_CORE[portfolio]
        CAL_CORE[calibration]
    end

    subgraph level5 [Level 5: Risk and Execution]
        RISK_VAR[risk.var]
        RISK_SENS[risk.sensitivities]
        RISK_SCEN[risk.scenarios]
        BT_ENGINE[backtesting]
        STR_ENGINE[streaming]
    end

    MD_CORE --> MD_CURVES
    MD_CORE --> MD_SURFACES
    MD_CORE --> MD_PROVIDERS
    MD_CORE --> MD_SCENARIOS

    MD_CORE --> INST_FX
    MD_CORE --> INST_EQ
    MD_CORE --> INST_IR

    MOD_COMMON --> MOD_ANALYTIC
    MOD_COMMON --> MOD_STOCHVOL
    MOD_COMMON --> MOD_NUMERIC
    MOD_COMMON --> MOD_PAYOFFS

    INST_FX --> PRC_FX
    INST_EQ --> PRC_EQ
    INST_IR --> PRC_IR
    MD_CURVES --> PRC_FX
    MD_SURFACES --> PRC_FX
    MOD_ANALYTIC --> PRC_FX
    MOD_NUMERIC --> PRC_FX

    PRC_FX --> PRC_REG
    PRC_EQ --> PRC_REG
    PRC_IR --> PRC_REG

    PRC_REG --> PORT_CORE
    MD_SURFACES --> CAL_CORE
    MOD_STOCHVOL --> CAL_CORE

    PORT_CORE --> RISK_VAR
    PORT_CORE --> RISK_SENS
    PORT_CORE --> RISK_SCEN
    MD_SCENARIOS --> RISK_SCEN

    MD_PROVIDERS --> BT_ENGINE
    PORT_CORE --> BT_ENGINE

    MD_PROVIDERS --> STR_ENGINE
    PORT_CORE --> STR_ENGINE
```

---

## 3. Pricing Flow

How an instrument gets priced: from instrument definition through market data lookup to model calculation.

```mermaid
flowchart LR
    subgraph instrument [Instrument Definition]
        TRADE[FxVanillaEuropeanOption]
        SPOT_ID[spot_id: MarketId]
        VOL_ID[vol_id: MarketId]
        CURVE_ID[curve_id: MarketId]
    end

    subgraph market [Market Data]
        MKT[Market Snapshot]
        QUOTE[Quote: spot=1.08]
        CURVE[Curve: df, zero_rate]
        VOLSURF[VolSurface: implied_vol]
    end

    subgraph pricer [Pricer Layer]
        REG[PricerRegistry]
        PRC[FxVanillaEuropeanOptionBsmPricer]
    end

    subgraph model [Model Layer]
        BSM[bsm_vanilla_price]
        GREEKS[bsm_vanilla_greeks]
    end

    subgraph result [Output]
        PV[PV: 42,350.00]
        DELTA[Delta: 0.55]
        GAMMA[Gamma: 0.02]
        VEGA[Vega: 15,000]
    end

    TRADE --> REG
    REG -->|resolve| PRC

    SPOT_ID --> MKT
    VOL_ID --> MKT
    CURVE_ID --> MKT

    MKT --> QUOTE
    MKT --> CURVE
    MKT --> VOLSURF

    PRC -->|extract params| BSM
    QUOTE --> BSM
    CURVE --> BSM
    VOLSURF --> BSM

    BSM --> PV
    PRC -->|greeks| GREEKS
    GREEKS --> DELTA
    GREEKS --> GAMMA
    GREEKS --> VEGA
```

---

## 4. Portfolio Pricing Flow

How a portfolio of instruments gets priced and aggregated.

```mermaid
flowchart TB
    subgraph input [Input]
        PORT[Portfolio]
        POS1[Position 1: FX Option]
        POS2[Position 2: IR Swap]
        POS3[Position 3: Equity Forward]
        MKT[Market]
    end

    subgraph pricing [Portfolio Pricer]
        PP[PortfolioPricer]
        REG[PricerRegistry]
        FX_PRC[FxVanillaBsmPricer]
        IR_PRC[IrSwapPricer]
        EQ_PRC[EquityForwardPricer]
    end

    subgraph results [Results]
        PR[PortfolioResult]
        POSRES[PositionResults]
        TOTALS[PortfolioTotals]
        AGG_PV[Total PV]
        AGG_DELTA[Aggregated Greeks]
    end

    PORT --> POS1
    PORT --> POS2
    PORT --> POS3

    POS1 --> PP
    POS2 --> PP
    POS3 --> PP
    MKT --> PP

    PP --> REG
    REG -->|resolve| FX_PRC
    REG -->|resolve| IR_PRC
    REG -->|resolve| EQ_PRC

    FX_PRC --> POSRES
    IR_PRC --> POSRES
    EQ_PRC --> POSRES

    POSRES --> PR
    PR --> TOTALS
    TOTALS --> AGG_PV
    TOTALS --> AGG_DELTA
```

---

## 5. Risk Computation Flow

How risk metrics (VaR, sensitivities, scenarios) are computed.

```mermaid
flowchart TB
    subgraph input [Input]
        PORT[Portfolio]
        MKT[Base Market]
        CFG[Risk Config]
    end

    subgraph sensitivities [Sensitivities Engine]
        SENS_ENG[compute_sensitivities]
        INFER[infer_risk_factors]
        BUMP[bump_and_reprice]
        ANALYTIC[analytic_greeks]
    end

    subgraph scenarios [Scenario Engine]
        SCEN_RUN[run_portfolio_scenarios]
        SHOCKS[ScenarioShock]
        SPOT_SHOCK[SpotShock]
        VOL_SHOCK[VolShock]
        RATE_SHOCK[ParallelRateShock]
        APPLY[apply_shock]
        REPRICE[reprice_portfolio]
    end

    subgraph var [VaR Engine]
        VAR_RUN[compute_var]
        HIST_VAR[HistoricalVaR]
        PARAM_VAR[ParametricVaR]
        MC_VAR[MonteCarloVaR]
    end

    subgraph output [Output]
        SENS_REP[SensitivitiesReport]
        SCEN_REP[ScenarioResult]
        VAR_RES[VarResult]
        ATTR_REP[AttributionReport]
    end

    PORT --> SENS_ENG
    MKT --> SENS_ENG
    SENS_ENG --> INFER
    INFER --> BUMP
    INFER --> ANALYTIC
    BUMP --> SENS_REP
    ANALYTIC --> SENS_REP

    PORT --> SCEN_RUN
    MKT --> SCEN_RUN
    SHOCKS --> SCEN_RUN
    SPOT_SHOCK --> SHOCKS
    VOL_SHOCK --> SHOCKS
    RATE_SHOCK --> SHOCKS
    SCEN_RUN --> APPLY
    APPLY --> REPRICE
    REPRICE --> SCEN_REP

    CFG --> VAR_RUN
    PORT --> VAR_RUN
    MKT --> VAR_RUN
    VAR_RUN --> HIST_VAR
    VAR_RUN --> PARAM_VAR
    VAR_RUN --> MC_VAR
    HIST_VAR --> VAR_RES
    PARAM_VAR --> VAR_RES
    MC_VAR --> VAR_RES

    SCEN_REP --> ATTR_REP
    SENS_REP --> ATTR_REP
```

---

## 6. Calibration Flow

How models are calibrated to market data.

```mermaid
flowchart LR
    subgraph market_data [Market Data]
        VOL_QUOTES[Market Vol Quotes]
        STRIKES[Strike Grid]
        EXPIRIES[Expiry Grid]
        FWD[Forward Price]
    end

    subgraph calibration [Calibration Engine]
        CAL_ENG[CalibrationEngine]
        OBJ[Objective Function]
        OPT[Optimizer]
        LBFGS[L-BFGS-B]
        LM[Levenberg-Marquardt]
        DE[Differential Evolution]
    end

    subgraph model_specific [Model-Specific Calibration]
        SABR_CAL[calibrate_sabr_to_smile]
        HESTON_CAL[calibrate_heston]
        HW_CAL[calibrate_hull_white]
        DUPIRE_CAL[calibrate_dupire]
    end

    subgraph output [Calibrated Parameters]
        SABR_PARAMS[SabrParameters]
        HESTON_PARAMS[HestonParameters]
        HW_PARAMS[HullWhiteParameters]
        LOCAL_VOL[LocalVolSurface]
    end

    VOL_QUOTES --> OBJ
    STRIKES --> OBJ
    EXPIRIES --> OBJ
    FWD --> OBJ

    OBJ --> CAL_ENG
    CAL_ENG --> OPT
    OPT --> LBFGS
    OPT --> LM
    OPT --> DE

    VOL_QUOTES --> SABR_CAL
    STRIKES --> SABR_CAL
    SABR_CAL --> SABR_PARAMS

    VOL_QUOTES --> HESTON_CAL
    HESTON_CAL --> HESTON_PARAMS

    VOL_QUOTES --> DUPIRE_CAL
    DUPIRE_CAL --> LOCAL_VOL
```

---

## 7. Backtesting Flow

How strategies are backtested on historical data.

```mermaid
flowchart TB
    subgraph data [Historical Data]
        PROVIDER[MarketDataProvider]
        HIST[HistoricalProvider]
        CSV[CsvDataProvider]
        DICT[DictDataProvider]
        ADAPTER[BacktestDataAdapter]
    end

    subgraph engine [Backtest Engine]
        BT_ENG[BacktestEngine]
        CFG[BacktestConfig]
        REPLAY[Replay Loop]
        STRATEGY[Strategy Function]
        PORT_STATE[PortfolioState]
    end

    subgraph execution [Execution Simulation]
        ORDERS[Orders]
        FILLS[Order Fills]
        COSTS[Transaction Costs]
        SLIPPAGE[Slippage]
    end

    subgraph metrics [Performance Metrics]
        RETURNS[Return Series]
        SHARPE[Sharpe Ratio]
        SORTINO[Sortino Ratio]
        DRAWDOWN[Max Drawdown]
        CALMAR[Calmar Ratio]
        ATTR[PnL Attribution]
    end

    subgraph output [Output]
        RESULT[BacktestResult]
        PERF[PerformanceMetrics]
        TRADES[Trade Records]
        PORT_VALUE[Portfolio Value Series]
    end

    HIST --> PROVIDER
    CSV --> PROVIDER
    DICT --> PROVIDER
    PROVIDER --> ADAPTER

    ADAPTER --> BT_ENG
    CFG --> BT_ENG

    BT_ENG --> REPLAY
    REPLAY -->|each date| STRATEGY
    STRATEGY -->|market, portfolio, context| ORDERS

    ORDERS --> FILLS
    FILLS --> COSTS
    COSTS --> SLIPPAGE
    SLIPPAGE --> PORT_STATE

    PORT_STATE --> PORT_VALUE
    PORT_VALUE --> RETURNS

    RETURNS --> SHARPE
    RETURNS --> SORTINO
    RETURNS --> DRAWDOWN
    RETURNS --> CALMAR
    PORT_STATE --> ATTR

    RETURNS --> RESULT
    SHARPE --> PERF
    SORTINO --> PERF
    DRAWDOWN --> PERF
    RESULT --> TRADES
```

---

## 8. Streaming / Live Trading Flow

How strategies execute on live or paper markets.

```mermaid
flowchart TB
    subgraph data_stream [Market Stream]
        STREAM_PROV[StreamingMarketDataProtocol]
        REPLAY_STREAM[ReplayStreamProvider]
        LIVE_FEED[LiveFeed Adapter]
    end

    subgraph engine [Streaming Engine]
        STR_ENG[StreamingEngine]
        CONTEXT[LiveContext]
        STRATEGY[Strategy Function]
        PORT_STATE[PortfolioState]
    end

    subgraph brokerage [Brokerage Layer]
        BROKER[BrokerageAdapter Protocol]
        PAPER[PaperBrokerageAdapter]
        LIVE[LiveBrokerageAdapter]
        SUBMIT[submit_order]
        FILL[apply_market]
    end

    subgraph output [Output]
        RESULT[StreamingRunResult]
        FILLS[Fill Records]
        FINAL_PORT[Final Portfolio]
    end

    REPLAY_STREAM --> STREAM_PROV
    LIVE_FEED --> STREAM_PROV

    STREAM_PROV --> STR_ENG
    STR_ENG --> CONTEXT

    CONTEXT -->|market, portfolio, context| STRATEGY
    STRATEGY --> ORDERS[Orders]

    ORDERS --> BROKER
    PAPER --> BROKER
    LIVE --> BROKER
    BROKER --> SUBMIT
    SUBMIT --> FILL
    FILL --> PORT_STATE

    PORT_STATE --> FINAL_PORT
    FILL --> FILLS
    FINAL_PORT --> RESULT
    FILLS --> RESULT
```

---

## 9. Orchestrator Pipeline Flow

How pipelines coordinate multiple steps and modules.

```mermaid
flowchart TB
    subgraph config [Configuration]
        RUN_CFG[RunConfig YAML/dict]
        VALIDATE[Validate Config]
    end

    subgraph registry [Pipeline Registry]
        REG[PipelineRegistry]
        PRICING_PIPE[price_portfolio Pipeline]
        RISK_PIPE[run_scenarios Pipeline]
        MD_PIPE[build_timeseries Pipeline]
    end

    subgraph context [Execution Context]
        CTX[Context]
        STATE[ctx.state]
        LOGGER[ctx.logger]
        ARTIFACTS[ArtifactStore]
    end

    subgraph steps [Pipeline Steps]
        STEP1[Step 1: Load Data]
        STEP2[Step 2: Build Provider]
        STEP3[Step 3: Price Portfolio]
        STEP4[Step 4: Run Scenarios]
        STEP5[Step 5: Save Results]
    end

    subgraph runner [Pipeline Runner]
        RUNNER[PipelineRunner]
        EXEC[Execute Steps]
    end

    subgraph output [Output]
        MANIFEST[RunManifest]
        CSV_OUT[CSV Files]
        JSON_OUT[JSON Files]
    end

    RUN_CFG --> VALIDATE
    VALIDATE --> REG
    REG --> PRICING_PIPE
    REG --> RISK_PIPE
    REG --> MD_PIPE

    CTX --> STATE
    CTX --> LOGGER
    CTX --> ARTIFACTS

    PRICING_PIPE --> RUNNER
    CTX --> RUNNER

    RUNNER --> EXEC
    EXEC --> STEP1
    STEP1 -->|write state| STATE
    EXEC --> STEP2
    STEP2 -->|read/write state| STATE
    EXEC --> STEP3
    STEP3 -->|read/write state| STATE
    EXEC --> STEP4
    STEP4 -->|read/write state| STATE
    EXEC --> STEP5
    STEP5 -->|write artifacts| ARTIFACTS

    ARTIFACTS --> CSV_OUT
    ARTIFACTS --> JSON_OUT
    ARTIFACTS --> MANIFEST
```

---

## 10. ML Pipeline Flow

How the ML framework trains, evaluates, and deploys models.

```mermaid
flowchart TB
    subgraph data_prep [Data Preparation]
        MC_DATA[MC Paths]
        ANALYTIC_DATA[Analytic Pricer]
        PORTFOLIO_DATA[Portfolio/GNN Data]
        PRICING_DS[build_pricing_dataset_from_mc]
        CAL_DS[build_calibration_dataset]
        GNN_DS[build_gnn_dataset_from_portfolio]
    end

    subgraph training [Training Pipeline]
        TRAINABLE[Trainable Protocol]
        KERAS_ADAPT[KerasTrainableAdapter]
        TRAIN_CFG[TrainingConfig]
        TRAIN_LOOP[TrainingLoop]
        RUN_TRAIN[run_training]
        CHECKPOINT[Checkpointing]
        EARLY_STOP[Early Stopping]
    end

    subgraph evaluation [Evaluation]
        EVAL_MODEL[evaluate_model]
        METRICS[Metrics: MSE, MAE, R2]
        BENCHMARK[Benchmark Comparison]
        EVAL_RESULT[EvaluationResult]
    end

    subgraph inference [Inference Pipeline]
        SAVE[save_model]
        LOAD[load_model]
        PREDICT[predict]
        ARTIFACT[Artifact Directory]
    end

    subgraph models [ML Models]
        NN_PRICER[NN Pricer]
        CAL_NET[Calibration Net]
        GNN_RNN[GNN-RNN Hybrid]
    end

    subgraph output [Output]
        TRAIN_RESULT[TrainingResult]
        PREDICTIONS[Predictions]
        DEPLOYED[Deployed Model]
    end

    MC_DATA --> PRICING_DS
    ANALYTIC_DATA --> PRICING_DS
    PORTFOLIO_DATA --> GNN_DS
    PRICING_DS --> TRAIN_LOOP
    CAL_DS --> TRAIN_LOOP
    GNN_DS --> TRAIN_LOOP

    TRAINABLE --> KERAS_ADAPT
    KERAS_ADAPT --> TRAIN_LOOP
    TRAIN_CFG --> TRAIN_LOOP
    RUN_TRAIN --> TRAIN_LOOP
    TRAIN_LOOP --> CHECKPOINT
    TRAIN_LOOP --> EARLY_STOP
    TRAIN_LOOP --> TRAIN_RESULT

    TRAIN_RESULT --> EVAL_MODEL
    EVAL_MODEL --> METRICS
    EVAL_MODEL --> BENCHMARK
    METRICS --> EVAL_RESULT
    BENCHMARK --> EVAL_RESULT

    TRAIN_RESULT --> SAVE
    SAVE --> ARTIFACT
    ARTIFACT --> LOAD
    LOAD --> PREDICT
    PREDICT --> PREDICTIONS

    NN_PRICER --> DEPLOYED
    CAL_NET --> DEPLOYED
    GNN_RNN --> DEPLOYED
```

---

## 11. Models and Numeric Methods

How analytic and numeric methods relate to each other.

```mermaid
flowchart TB
    subgraph analytic [Analytic Models]
        BSM[Black-Scholes-Merton]
        B76[Black76]
        BACH[Bachelier]
    end

    subgraph stochvol [Stochastic Volatility]
        HESTON[Heston]
        SABR[SABR]
    end

    subgraph shortrate [Short Rate Models]
        HW[Hull-White]
        BK[Black-Karasinski]
    end

    subgraph jump [Jump Models]
        MERTON[Merton Jump-Diffusion]
        VG[Variance Gamma]
    end

    subgraph numeric [Numeric Methods]
        MC[Monte Carlo]
        FDE[Finite Difference]
        QMC[Quasi-Monte Carlo]
        LSM[LSM for American]
    end

    subgraph payoffs [Payoff Layer]
        VANILLA[VanillaPayoff]
        DIGITAL[DigitalPayoff]
        BARRIER[BarrierPayoff]
        ASIAN[AsianPayoff]
        LOOKBACK[LookbackPayoff]
        TOUCH[TouchPayoff]
    end

    subgraph dynamics [Dynamics]
        GBM[GBM Dynamics]
        HESTON_DYN[Heston Dynamics]
        HW_DYN[Hull-White Dynamics]
    end

    BSM --> VANILLA
    BSM --> DIGITAL
    B76 --> VANILLA

    HESTON --> HESTON_DYN
    HW --> HW_DYN
    GBM --> MC

    HESTON_DYN --> MC
    HW_DYN --> MC
    HW_DYN --> FDE

    MC --> VANILLA
    MC --> BARRIER
    MC --> ASIAN
    MC --> LOOKBACK
    MC --> TOUCH

    FDE --> VANILLA
    FDE --> BARRIER

    LSM --> VANILLA
```

---

## 12. Complete Data Flow: Instrument to Report

End-to-end flow from trade definition to risk report.

```mermaid
flowchart TB
    subgraph definition [Definition Phase]
        INST[Define Instrument]
        MKT_IDS[Set MarketIds]
        PORT[Build Portfolio]
    end

    subgraph market [Market Phase]
        MD_PROV[MarketDataProvider]
        MKT_SNAP[Market Snapshot]
        CURVES[Curves]
        VOLS[Vol Surfaces]
        QUOTES[Quotes]
    end

    subgraph pricing [Pricing Phase]
        REG[PricerRegistry]
        PRC[Pricer]
        PORT_PRC[PortfolioPricer]
        PV[PV]
        GREEKS[Greeks]
    end

    subgraph risk [Risk Phase]
        SCEN[Scenarios]
        SHOCK[Apply Shocks]
        VAR[VaR]
        SENS[Sensitivities]
    end

    subgraph reporting [Reporting Phase]
        RISK_REP[RiskReport]
        VAR_SUM[VarSummary]
        SCEN_REP[ScenarioReport]
        PLOTS[Risk Plots]
    end

    INST --> MKT_IDS
    MKT_IDS --> PORT

    MD_PROV --> MKT_SNAP
    MKT_SNAP --> CURVES
    MKT_SNAP --> VOLS
    MKT_SNAP --> QUOTES

    PORT --> PORT_PRC
    MKT_SNAP --> PORT_PRC
    REG --> PORT_PRC
    PORT_PRC --> PRC
    PRC --> PV
    PRC --> GREEKS

    PORT --> SCEN
    MKT_SNAP --> SCEN
    SCEN --> SHOCK
    SHOCK --> RISK_REP

    PORT --> VAR
    MKT_SNAP --> VAR
    VAR --> VAR_SUM

    PORT --> SENS
    MKT_SNAP --> SENS
    SENS --> RISK_REP

    RISK_REP --> SCEN_REP
    VAR_SUM --> RISK_REP
    RISK_REP --> PLOTS
```

---

## Summary Table: Module Responsibilities

| Module | Primary Responsibility | Key Interfaces |
|--------|----------------------|----------------|
| **marketdata** | Market data infrastructure | `Market`, `MarketId`, `Curve`, `VolSurface`, `MarketDataProvider` |
| **instruments** | Trade/product definitions | `FxVanillaEuropeanOption`, `IrSwap`, etc. (dataclasses) |
| **models** | Mathematical pricing models | `bsm_price()`, `HestonDynamics`, `GbmDynamics`, payoffs |
| **pricers** | Pricing adapters | `InstrumentPricer` protocol, `PricerRegistry` |
| **portfolio** | Portfolio management | `Portfolio`, `Position`, `PortfolioPricer` |
| **risk** | Risk computation | `compute_var()`, `compute_sensitivities()`, `run_portfolio_scenarios()` |
| **calibration** | Model calibration | `CalibrationEngine`, `calibrate_sabr_to_smile()` |
| **backtesting** | Historical strategy testing | `BacktestEngine`, `PerformanceMetrics` |
| **streaming** | Live/paper trading | `StreamingEngine`, `BrokerageAdapter` |
| **orchestrator** | Workflow coordination | `Pipeline`, `PipelineRunner`, `Context` |
| **machine_learning** | ML training/inference | `Trainable`, `run_training()`, `evaluate_model()`, `predict()` |
| **core** | Utilities and plotting | `style`, `utils`, risk/marketdata/pricer plots |
| **ui** | Dash UIs | `create_app()`, shared components |

---

## Dependency Summary

```
Level 0: marketdata.core, instruments.core, models.common
    ↓
Level 1: marketdata (curves, surfaces, providers, scenarios)
    ↓
Level 2: instruments (fx, equity, ir), models (analytic, numeric, dynamics, payoffs)
    ↓
Level 3: pricers (fx, equity, ir, registry)
    ↓
Level 4: portfolio, calibration
    ↓
Level 5: risk, backtesting, streaming, machine_learning
    ↓
Level 6: orchestrator, ui
```
