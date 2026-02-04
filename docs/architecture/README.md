# QuantStrata Architecture Documentation

This section provides visual and detailed documentation of the QuantStrata library architecture.

---

## Documents

| Document | Description |
|----------|-------------|
| [**QuantStrata Library Overview**](QuantStrataLibraryOverview.md) | **Start here.** Single-document architecture overview for quants: layers, components, workflows, pipelines, and ecosystem |
| [Ecosystem Diagrams](ecosystem_diagrams.md) | Visual diagrams showing module interactions, data flows, and component relationships |
| [Component Reference](component_reference.md) | Detailed reference of what each module provides and requires |
| [Orchestrator Pipeline Documentation](orchestrator_pipeline_documentation.md) | Pipeline reference, state keys, configuration, and best practices |

---

## Quick Overview

QuantStrata is organised into layers:

1. **Foundation Layer** — `marketdata`, `instruments`, `models`
   - Market data infrastructure (curves, surfaces, providers)
   - Financial instrument definitions
   - Mathematical pricing models

2. **Computation Layer** — `pricers`, `calibration`
   - Pricing adapters connecting instruments + market to models
   - Model calibration to market data

3. **Execution Layer** — `portfolio`, `risk`, `backtesting`, `streaming`, `machine_learning`
   - Portfolio management and pricing
   - Risk computation (VaR, sensitivities, scenarios)
   - Historical backtesting and live trading
   - ML training and inference

4. **Orchestration Layer** — `orchestrator`
   - Pipeline coordination
   - Artifact management
   - CLI interface

5. **Interface Layer** — `ui`, `core/reporting`
   - Dash UIs
   - Plotting and export utilities

---

## Key Design Principles

1. **Separation of Concerns**
   - Instruments = product definitions (no pricing logic)
   - Models = pure math functions (no market objects)
   - Pricers = adapters (market → model parameters)
   - Portfolio = aggregation (routes instruments to pricers)

2. **Protocol-Based Interfaces**
   - `Curve`, `VolSurface`, `MarketDataProvider` protocols
   - `InstrumentPricer`, `Trainable` protocols
   - Duck typing with runtime_checkable

3. **Immutability**
   - Core objects are frozen dataclasses
   - `Market`, `MarketId`, `Portfolio` are immutable
   - Thread-safe by design

4. **Registry Pattern**
   - `PricerRegistry` for instrument → pricer routing
   - `PipelineRegistry` for pipeline discovery
   - Extensible without modifying core code

5. **Layered Dependencies**
   - Foundation modules have no internal dependencies
   - Higher layers depend only on lower layers
   - Clear dependency graph
