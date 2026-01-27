# FX Market Data Examples

This directory contains a comprehensive suite of examples demonstrating the end-to-end FX market data workflow using the QuantStrata marketdata core. These examples are designed to mirror real front-office quant desk workflows, providing deterministic runs, clear visualizations, and clean outputs that integrate directly with pricing and risk systems.

## Overview

The examples follow a logical progression from building timeseries datasets to applying scenario shocks and persisting data:

1. **Build Timeseries Dataset** - Construct multi-product FX market datasets with scenarios
2. **Get Market Snapshot** - Extract pricing-ready market snapshots from datasets
3. **Apply Scenario Shocks** - Apply deterministic shocks to market data
4. **Save and Reload Dataset** - Persist and reload market datasets
5. **Build FX Smile from Quotes** - Construct volatility surfaces from market quotes

## Design Principles

These examples enforce key architectural principles of the QuantStrata library:

- **Provider Agnosticism**: All examples use the provider factory pattern, allowing seamless switching between `SyntheticProvider`, `StaticProvider`, and future API providers
- **Market Boundary**: Pricing and risk systems consume only `Market` snapshots, never provider internals
- **Determinism**: All examples use seeded random number generation for reproducible results
- **Professional Output**: Plots and diagnostics match front-office desk standards

## Prerequisites

- A market data provider available (default is `SyntheticProvider`)
- Understanding of `MarketId` naming conventions (e.g., `FX.SPOT.EURUSD`, `FX.VOL.EURUSD|cut=NY,convention=delta25`)
- Python environment with QuantStrata dependencies installed

---

## Example 1: Build Timeseries Dataset

**Script**: `01_build_timeseries_dataset.py`

### Functionality

This example demonstrates the canonical workflow for building a multi-product FX **`MarketDataset`** (SPOT + CURVE + VOL) across a date range and scenario set. It showcases:

- Building a provider-agnostic market data provider
- Requesting timeseries datasets with multiple scenarios
- Running desk-standard diagnostics (fan charts, returns, correlations)
- Extracting pricing-ready **`Market`** snapshots
- Visualizing market objects (quotes, curves, volatility surfaces)
- Comparing scenario snapshots to validate dispersion

This script is the **foundational example** demonstrating the marketdata core contract:
- **Providers** produce **`MarketDataset`** (time × scenarios) for plotting, backtests, and scenario analysis
- **Pricing/Risk** consumes **`Market`** snapshots only (provider-agnostic boundary)

### Walkthrough

#### Step 1: Define Market Universe

The example defines a realistic FX universe focused on volatility-driven instruments. In FX, **VOL is dependency-rich**, so requesting VOL surfaces triggers automatic dependency closure:

- `FX.VOL.<PAIR>` requires `FX.SPOT.<PAIR>`
- If `dom/for` qualifiers exist, requires `IR.CURVE.<CCY>.OIS` for each currency

```python
vol_ids = [
    _fx_vol_id("EURUSD", cut="NY", convention="delta25", dom="USD", foreign="EUR"),
    _fx_vol_id("GBPUSD", cut="NY", convention="delta25", dom="USD", foreign="GBP"),
    _fx_vol_id("USDJPY", cut="TK", convention="delta25", dom="JPY", foreign="USD"),
]

curve_ids = [
    _ir_curve_id("USD"),
    _ir_curve_id("EUR"),
    _ir_curve_id("GBP"),
    _ir_curve_id("JPY"),
]

universe = Universe(ids=vol_ids + curve_ids)
```

**Key Design**: The provider performs automatic dependency closure, so we only need to request VOL surfaces explicitly. The provider will automatically include required SPOT rates and interest rate curves.

#### Step 2: Build Provider with Realistic Configuration

The example builds a `SyntheticProvider` with pair-specific overrides to ensure market-realistic behavior:

```python
provider = _build_provider_with_realistic_overrides(vol_ids=vol_ids)
```

**Why Overrides Matter**: Default specs are tuned for EURUSD-like scales (spot ~1.10, strikes ~0.9..1.2). USDJPY requires different scales (spot ~110, strikes ~90..130). The override mechanism allows per-instrument calibration.

**Configuration Details**:
- **EURUSD/GBPUSD**: Spot ~1.10, strikes [0.90, 1.00, 1.10, 1.20], ATM vol ~12%
- **USDJPY**: Spot ~110, strikes [90, 100, 110, 120, 130], ATM vol ~11%
- Deterministic seed (7) for reproducible results

#### Step 3: Request Timeseries Dataset

The example constructs a `TimeseriesRequest` and retrieves a `MarketDataset`:

```python
request = TimeseriesRequest(
    start="2026-01-01",
    end="2026-02-01",
    freq="D",
    universe=universe,
    scenarios=50,
)

dataset = provider.get_timeseries(request)
```

**Dataset Structure**:
- **Dates**: List of date strings covering the requested range
- **Panels**: Dictionary of `MarketId` → `Panel` for quote data (shape [T, S] where T=time, S=scenarios)
- **Curve Parameters**: Dictionary of `MarketId` → `Panel` for curve parameters (tenors + zero rates)
- **Vol Parameters**: Dictionary of `MarketId` → `Panel` for vol surface parameters (expiry × strike grids)
- **Factories**: Reconstruction functions to build `Curve` and `VolSurface` objects from parameters

#### Step 4: Timeseries Diagnostics

The example generates three types of timeseries diagnostics:

**4a) Fan Charts**: Visualize spot path dispersion across scenarios using quantile-based fan charts:

```python
plot_spot_fan_chart(
    dates=dataset.dates,
    spot_paths=spot_paths,  # [T, S] array
    title=f"{pair} spot fan chart",
    fan=FanSpec(q_low=0.05, q_mid=0.50, q_high=0.95, max_scenario_lines=8),
)
```

**4b) Log Returns**: Plot median log returns across scenarios to assess drift and volatility:

```python
plot_log_return_timeseries(
    dates=dataset.dates,
    spot_paths=spot_paths,
    title=f"{pair} median log returns",
)
```

**4c) Correlation Heatmap**: Compute and visualize cross-pair return correlations:

```python
returns_by_pair = {pair: flatten_log_returns_all_scenarios(spot_paths) for ...}
plot_return_correlation_heatmap(
    returns_by_label=returns_by_pair,
    title="FX spot log-return correlation",
)
```

#### Step 5: Extract Market Snapshot

The example extracts a pricing-ready `Market` snapshot at a specific time and scenario:

```python
market = dataset.snapshot(time_idx=-1, scenario_idx=0)  # Last date, base scenario
```

**Snapshot Process**:
1. Slices quote panels at `[time_idx, scenario_idx]` → `Quote` objects
2. Slices curve parameter panels → reconstructs `Curve` objects via factories
3. Slices vol parameter panels → reconstructs `VolSurface` objects via factories
4. Returns immutable `Market` object with `asof` date

**Market Object Structure**:
- `asof`: Date string of the snapshot
- `quotes`: `Mapping[MarketId, Quote]` - scalar quote values
- `curves`: `Mapping[MarketId, Curve]` - interest rate curve objects
- `vols`: `Mapping[MarketId, VolSurface]` - volatility surface objects

#### Step 6: Visualize Market Objects

The example generates desk-standard visualizations for each market object type:

**6a) Quotes**: Bar chart of spot rates:

```python
plot_quotes(spot_quotes, title=f"FX Spots @ {market.asof}")
```

**6b) Curves**: Three views of interest rate curves:
- Discount factors: `plot_curve_df(curve, t_max=10.0)`
- Zero rates: `plot_curve_zero_rate(curve, t_max=10.0)`
- Forward rates: `plot_curve_forward_rate(curve, t1=0.5, t2_max=10.0)`

**6c) Volatility Surfaces**: Three views of vol surfaces:
- Heatmap: `plot_vol_surface_heatmap(surface)`
- Smile slices: `plot_vol_smile_slices(surface)` - vol vs strike at fixed expiries
- 3D surface: `plot_vol_surface(surface)` - interactive 3D visualization

#### Step 7: Scenario Comparison

The example compares snapshots across scenarios to validate dispersion:

```python
base = dataset.snapshot(time_idx=time_idx, scenario_idx=0)
shocked = {
    "SCEN_1": dataset.snapshot(time_idx=time_idx, scenario_idx=1),
    "SCEN_10": dataset.snapshot(time_idx=time_idx, scenario_idx=10),
}
```

**Comparison Visualizations**:
- Spot comparison: `plot_spot_comparison(base, shocked, spot_id)`
- Curve comparison: `plot_curve_df_comparison(base, shocked, curve_id, times)`
- Vol comparison: `plot_vol_comparison(base, shocked, vol_id, expiry, strikes)`

### Required Inputs

**Configuration** (`ExampleConfig`):
- `start`: Start date (ISO format, e.g., "2026-01-01")
- `end`: End date (ISO format, e.g., "2026-02-01")
- `freq`: Date frequency ("D" for daily, "W" for weekly, etc.)
- `n_scenarios`: Number of scenarios to generate (default: 50)
- `snapshot_time_idx`: Time index for snapshot extraction (default: -1, last date)
- `snapshot_scenario_idx`: Scenario index for snapshot extraction (default: 0, base scenario)
- `out_dir_name`: Output directory for saved plots (default: "outputs/01_build_timeseries_dataset")

**Market Universe**:
- FX VOL MarketIds with qualifiers (cut, convention, dom, for)
- IR CURVE MarketIds (optional, but recommended for explicitness)

**Provider Configuration**:
- Seed for deterministic generation (default: 7)
- Pair-specific overrides for spot levels, strike grids, and vol parameters

### Technical Description

#### Core Components

**`MarketDataset`** (`src/marketdata/core/dataset.py`):
- **Purpose**: Time-series container storing multi-scenario market data
- **Structure**:
  - `dates: List[str]` - Date grid
  - `n_scenarios: int` - Number of scenarios
  - `panels: Mapping[MarketId, Panel]` - Quote data panels [T, S]
  - `curve_params: Mapping[MarketId, Panel]` - Curve parameter panels [T, S, K, 2] (tenor, zero_rate)
  - `vol_params: Mapping[MarketId, Panel]` - Vol parameter panels [T, S, M, N] (expiry × strike)
  - `curve_factories: Mapping[MarketId, CurveFactory]` - Reconstruction functions
  - `vol_factories: Mapping[MarketId, VolSurfaceFactory]` - Reconstruction functions
- **Key Method**: `snapshot(time_idx, scenario_idx) -> Market` - Extract pricing snapshot

**`Market`** (`src/marketdata/core/market.py`):
- **Purpose**: Immutable market snapshot consumed by pricing and risk
- **Structure**:
  - `asof: str` - Snapshot date
  - `quotes: Mapping[MarketId, Quote]` - Scalar quote values
  - `curves: Mapping[MarketId, Curve]` - Interest rate curve objects
  - `vols: Mapping[MarketId, VolSurface]` - Volatility surface objects
- **Design Rule**: Pricers depend ONLY on `Market`, never on providers/files/APIs

**`SyntheticProvider`** (`src/marketdata/providers/synthetic/provider.py`):
- **Purpose**: Generate synthetic market data using stochastic models
- **Key Features**:
  - Dependency closure: Automatically includes prerequisites
  - Deterministic RNG: Per-MarketId substreams for reproducibility
  - Order-independent: Generation order doesn't affect values

#### Provider Factory Pattern

**`build_provider()`** (`src/marketdata/providers/factory.py`):
- **Purpose**: Provider-agnostic factory function
- **Usage**: `provider = build_provider(SyntheticProviderSpec(...))`
- **Benefit**: Calling code remains unchanged when switching providers

**`SyntheticProviderSpec`**:
- `seed: int` - Global seed for deterministic generation
- `config: SyntheticProviderConfig` - Provider-specific configuration
- `name: str` - Provider identifier

**`SyntheticProviderConfig`**:
- `spot_overrides: Mapping[MarketId, SpotGbmSpec]` - Per-instrument spot specs
- `vol_overrides: Mapping[MarketId, VolGridSmileSpec]` - Per-instrument vol specs
- `curve_method: str` - Curve generation method ("zeros" or "bootstrap")

#### Generator Specifications

**`SpotGbmSpec`** (`src/marketdata/providers/synthetic/specs.py`):
- **Model**: Geometric Brownian Motion (GBM)
- **Parameters**:
  - `initial_level: float` - Starting spot level
  - `drift: float` - Annual drift (default: 0.0)
  - `vol: float` - Annual volatility (default: 0.10)
  - `dt: float` - Time step (default: 1/252, daily)
  - `initial_dispersion: float` - Lognormal dispersion at t=0 (default: 0.0)
- **Generation**: `S(t+dt) = S(t) * exp((drift - 0.5*vol²)*dt + vol*sqrt(dt)*Z)` where Z ~ N(0,1)

**`VolGridSmileSpec`**:
- **Model**: Parametric smile in log-moneyness + term structure
- **Parameters**:
  - `expiries: np.ndarray` - Expiry grid (years)
  - `strikes: np.ndarray` - Strike grid (absolute strikes)
  - `atm_vol: float` - At-the-money volatility (default: 0.12)
  - `skew: float` - Volatility skew parameter (default: -0.15)
  - `smile: float` - Volatility smile parameter (default: 0.20)
  - `term: float` - Term structure parameter (default: 0.10)
  - `noise_scale: float` - Additive noise scale (default: 0.002)
- **Generation**: Parametric form combining log-moneyness and expiry effects

**`CurveZeroSpec`**:
- **Model**: Zero-rate curve with shape parameters
- **Parameters**:
  - `tenors: np.ndarray` - Tenor grid (years)
  - `base_rate: float` - Base interest rate (default: 0.02)
  - `slope: float` - Curve slope (default: 0.00)
  - `curvature: float` - Curve curvature (default: 0.00)
  - `noise_scale: float` - Additive noise scale (default: 0.0005)
- **Generation**: `r(t) = base_rate + slope*t + curvature*t² + noise`

### Underlying Methodology

#### Dependency Closure Algorithm

When a `MarketDataset` is requested, the provider performs automatic dependency closure:

1. **Initial Set**: Start with user-requested `MarketId`s
2. **Registry Lookup**: For each `MarketId`, query the generator registry for prerequisites
3. **Recursive Expansion**: Add prerequisites to the set, recursively expanding their prerequisites
4. **Fixed Point**: Continue until no new dependencies are found

**Example**: Requesting `FX.VOL.EURUSD|dom=USD,for=EUR` triggers:
- Add `FX.SPOT.EURUSD|dom=USD,for=EUR` (VOL requires SPOT)
- Add `IR.CURVE.USD.OIS|ccy=USD` (dom qualifier requires USD curve)
- Add `IR.CURVE.EUR.OIS|ccy=EUR` (for qualifier requires EUR curve)

#### Deterministic Random Number Generation

The synthetic provider ensures reproducibility through deterministic RNG:

1. **Global Seed**: Provider initialized with a seed (e.g., 7)
2. **Per-MarketId Substreams**: Each `MarketId` gets its own RNG substream derived from `(seed, MarketId.key())`
3. **Order Independence**: Generation order doesn't affect values because each `MarketId` uses an independent substream
4. **Scenario Independence**: Each scenario uses a separate substream derived from `(seed, MarketId.key(), scenario_idx)`

**Implementation**: Uses NumPy's `Generator` with a seed derived from hash of `(seed, MarketId.key())`.

#### Panel-Based Storage

Market data is stored in `Panel` objects with explicit axis naming:

- **Quote Panels**: Shape `[T, S]` where T=time, S=scenarios
  - Axis names: `["time", "scenario"]`
  - Data type: `float`
- **Curve Parameter Panels**: Shape `[T, S, K, 2]` where K=tenors
  - Axis names: `["time", "scenario", "tenor", "param"]`
  - Last dimension: `[tenor, zero_rate]`
- **Vol Parameter Panels**: Shape `[T, S, M, N]` where M=expiries, N=strikes
  - Axis names: `["time", "scenario", "expiry", "strike"]`
  - Data: Volatility values at each (expiry, strike) grid point

**Snapshot Extraction**: The `snapshot()` method slices panels at `[time_idx, scenario_idx]` and reconstructs pricing objects via factories.

#### Geometric Brownian Motion for Spot Generation

FX spot rates are generated using GBM:

```
dS(t) = S(t) * (μ * dt + σ * dW(t))
```

**Discretization** (Euler-Maruyama):
```
S(t+dt) = S(t) * exp((μ - 0.5*σ²)*dt + σ*√dt*Z)
```

Where:
- `μ` = drift (from `SpotGbmSpec.drift`)
- `σ` = volatility (from `SpotGbmSpec.vol`)
- `dt` = time step (from `SpotGbmSpec.dt`, default 1/252 for daily)
- `Z` ~ N(0,1) from deterministic RNG

**Multi-Scenario**: Each scenario uses an independent random walk, but all scenarios share the same initial level (unless `initial_dispersion > 0`).

#### Volatility Surface Generation

Volatility surfaces are generated using a parametric model:

**Base Volatility**:
```
σ_base(τ) = atm_vol + term * τ
```

**Moneyness Adjustment**:
```
log_moneyness = log(K / S(t))
σ_adjust = skew * log_moneyness + smile * log_moneyness²
```

**Final Volatility**:
```
σ(τ, K) = σ_base(τ) + σ_adjust + noise
```

Where:
- `τ` = time to expiry
- `K` = strike
- `S(t)` = current spot (from dependency)
- `noise` ~ N(0, noise_scale²) for realism

**Grid Generation**: The provider generates volatility values at each (expiry, strike) grid point specified in `VolGridSmileSpec`, storing them in vol parameter panels.

#### Curve Reconstruction

Curves are stored as parameter blocks and reconstructed via factories:

1. **Storage**: Curve parameters stored as `[tenor, zero_rate]` pairs in curve parameter panels
2. **Snapshot**: At snapshot time, slice parameter block at `[time_idx, scenario_idx]`
3. **Reconstruction**: `CurveFactory.build(params)` creates a `Curve` object
4. **Interpolation**: The `Curve` object provides interpolation methods (`df(t)`, `zero_rate(t)`, `forward_rate(t1, t2)`)

**Factory Pattern**: This allows efficient storage (parameters only) while providing rich curve objects at snapshot time.

---

## Example 2: Get Market Snapshot

**Script**: `02_get_market_snapshot.py`

### Functionality

[To be filled when script is implemented]

### Walkthrough

[To be filled when script is implemented]

### Required Inputs

[To be filled when script is implemented]

### Technical Description

[To be filled when script is implemented]

### Underlying Methodology

[To be filled when script is implemented]

---

## Example 3: Apply Scenario Shocks

**Script**: `03_apply_scenario_shocks.py`

### Functionality

[To be filled when script is implemented]

### Walkthrough

[To be filled when script is implemented]

### Required Inputs

[To be filled when script is implemented]

### Technical Description

[To be filled when script is implemented]

### Underlying Methodology

[To be filled when script is implemented]

---

## Example 4: Save and Reload Dataset

**Script**: `04_save_and_reload_dataset.py`

### Functionality

[To be filled when script is implemented]

### Walkthrough

[To be filled when script is implemented]

### Required Inputs

[To be filled when script is implemented]

### Technical Description

[To be filled when script is implemented]

### Underlying Methodology

[To be filled when script is implemented]

---

## Example 5: Build FX Smile from Quotes

**Script**: `05_build_fx_smile_from_quotes.py`

### Functionality

[To be filled when script is implemented]

### Walkthrough

[To be filled when script is implemented]

### Required Inputs

[To be filled when script is implemented]

### Technical Description

[To be filled when script is implemented]

### Underlying Methodology

[To be filled when script is implemented]

---

## Key Project Components

All examples use the following core components from the QuantStrata library:

### Identifiers
- **`MarketId`** (`src/marketdata/core/ids.py`): Structured identifier for market data (asset_class, mkt_type, name, qualifiers)

### Requests
- **`Universe`** (`src/marketdata/core/requests.py`): Collection of requested `MarketId`s
- **`TimeseriesRequest`** (`src/marketdata/core/requests.py`): Request for timeseries data (start, end, freq, universe, scenarios)
- **`MarketRequest`** (`src/marketdata/core/requests.py`): Request for single market snapshot (asof, universe, scenario)

### Containers
- **`MarketDataset`** (`src/marketdata/core/dataset.py`): Timeseries/scenario container
- **`Market`** (`src/marketdata/core/market.py`): Immutable pricing snapshot
- **`Panel`** (`src/marketdata/core/panel.py`): Multi-axis data container

### Provider Abstraction
- **`MarketDataProvider`** (`src/marketdata/providers/interfaces.py`): Protocol defining provider interface
- **`build_provider()`** (`src/marketdata/providers/factory.py`): Factory function for provider-agnostic construction

### Plotting
- **`src/core/reporting/plots/marketdata/quotes.py`**: Quote visualization
- **`src/core/reporting/plots/marketdata/curves.py`**: Curve visualization
- **`src/core/reporting/plots/marketdata/surfaces.py`**: Volatility surface visualization
- **`src/core/reporting/plots/marketdata/scenarios.py`**: Scenario comparison plots
- **`src/core/reporting/plots/marketdata/timeseries.py`**: Timeseries diagnostics

---

## Running the Examples

### Basic Usage

```bash
# Run example 1 (interactive mode, displays plots)
python examples/marketdata/fx/01_build_timeseries_dataset.py

# Run example 1 (save mode, writes plots to disk)
python -c "from examples.marketdata.fx.01_build_timeseries_dataset import main; main(save_files=True)"
```

### Configuration

Each example uses a configuration dataclass (`ExampleConfig`) that can be modified to adjust:
- Date ranges
- Scenario counts
- Snapshot selection
- Output directories

### Output

Examples generate two types of output:
1. **Console Output**: Dataset summaries, diagnostics, and status messages
2. **Visualizations**: Matplotlib figures (either displayed interactively or saved to disk)

---

## Best Practices

1. **Use Provider Factory**: Always use `build_provider()` to maintain provider-agnostic code
2. **Explicit MarketIds**: Use helper functions to construct `MarketId`s with proper qualifiers
3. **Dependency Closure**: Let the provider handle dependency closure automatically
4. **Snapshot Extraction**: Use `dataset.snapshot()` to get pricing-ready `Market` objects
5. **Deterministic Seeds**: Use fixed seeds for reproducible results in testing/development
6. **Error Handling**: Check for missing market data before plotting (use `market.has(mkt_id)`)

---

## Troubleshooting

### Common Issues

**Missing Market Data**:
- Ensure requested `MarketId`s match provider capabilities
- Check that qualifiers are correctly specified
- Verify dependency closure includes all prerequisites

**Plot Generation Errors**:
- Ensure market objects exist in snapshot (`market.has(mkt_id)`)
- Check that curve/vol factories are properly configured
- Verify date ranges and scenario indices are within bounds

**Provider Errors**:
- Verify provider configuration (specs, overrides)
- Check that seed is set for deterministic generation
- Ensure requested date ranges are valid

---

## Further Reading

- **Core Documentation**: See `docs/interfaces.md` for detailed API documentation
- **Provider Documentation**: See provider-specific documentation in `src/marketdata/providers/`
- **Plotting Documentation**: See plotting module docstrings for visualization options
