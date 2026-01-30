# Interfaces Contract (V1 → Vn)

This document **freezes the public interfaces** for instruments, payoffs, pricers, the market snapshot, and routing.
The goal is to **stop refactor churn**: adding a new product should not require changing existing pricers, tests, or
core plumbing.

**Design rule:** *Interfaces are stable; implementations evolve behind them.*

---

## 0) Scope and stability guarantees

### Frozen in V1 (do not break without a major version bump)
- `Market` snapshot getters (`quote`, `curve(...).df(t)`, `vol_surface(...).vol(expiry, strike)`)
- Instrument dataclasses for supported products (trade specs only, no pricing logic)
- Payoff protocols and base classes:
  - `TerminalPayoff1D` / `BasePayoff1D`
  - `PathPayoff1D` / `BasePathPayoff1D`
- Pricer API:
  - `price(instrument, market) -> float`
  - optional `greeks(instrument, market) -> dict[str, float]`
  - optional `diagnostics(instrument, market, ...) -> Any`
- Routing via `PricerRegistry.resolve(instrument, pricer_id=None|str)`

### Allowed to evolve (implementation-only)
- Numeric engines (MC variance reduction, FD schemes, grid builders, smoothing, etc.)
- Calibration builders (curves/surfaces)
- Performance (Numba/JAX/TF backends) **as long as public interfaces stay unchanged**

---

## 1) Directory conventions (V1)

Recommended layout (matches current repo intent):

```
src/
  instruments/
    fx/
      linear/
      options/
  marketdata/
    market.py
  models/
    payoffs/
      base.py
      types.py
      vanilla.py
      digital.py
      barrier.py            # (future) MC-only path payoff in V1.1+
    numeric/
      finite_difference/
      monte_carlo/
  pricers/
    registry.py
    fx/
      ...
docs/
  interfaces.md             # this file
```

---

## 2) Market snapshot contract

Pricers treat `Market` as an immutable snapshot at an **as-of** time.

### Required methods (V1)
- `market.quote(quote_id) -> float`
- `market.curve(curve_id).df(t: float) -> float`
- `market.vol_surface(vol_id).vol(expiry: float, strike: float) -> float`

### Conventions
- `t`, `expiry` are expressed in **year fractions** (ACT/365F or whatever your global convention is).
- Curve `df(t)` returns a discount factor to time `t`.
- Vol surface `vol(expiry, strike)` returns an **implied volatility** compatible with the model used in the pricer.

**Important:** Pricers are responsible for converting discount factors into continuous rates if needed.

---

## 3) Instrument contract

Instruments are **dumb dataclasses**: they contain *only* trade specification and market identifiers.

### Example (FX European vanilla)
Typical fields:
- `option_type: Literal["call","put"]`
- `notional: float`
- `strike: float`
- `expiry: float`
- `spot_id: MarketId`
- `vol_id: MarketId`
- `domestic_curve_id: MarketId`
- `foreign_curve_id: MarketId`

### Rules
- Instruments must not import pricers or numeric engines.
- Instruments may validate obvious invariants (e.g., non-negative notional, strike > 0), but keep this light.

---

## 4) Payoff contract (single source of truth)

Payoffs live under `src/models/payoffs/` and are the **single source of truth** for terminal and path-dependent payoff
evaluation.

### 4.1 Terminal (non path-dependent) payoff interface

Defined in `src/models/payoffs/base.py`:

- `TerminalPayoff1D` protocol
- `BasePayoff1D` concrete base class

#### Required methods/properties
- `terminal(spot: np.ndarray) -> np.ndarray`
- `intrinsic(spot: np.ndarray) -> np.ndarray` (default == terminal for European-style products)
- `is_path_dependent -> bool` (False)
- `__call__(spot)` alias for `terminal(spot)`

#### Where terminal payoffs are used
- **BSM / analytic**: payoff for interpretation / parity checks (not always required if closed form exists).
- **FD/PDE**: terminal condition `V(T,S) = payoff(S)`.

**Key principle:** FD/PDE only needs terminal payoff for 1D Markov products.

### 4.2 Path-dependent payoff interface

Defined in `src/models/payoffs/base.py`:

- `PathPayoff1D` protocol
- `BasePathPayoff1D` concrete base class

#### Required methods/properties
- `terminal_from_paths(paths: np.ndarray) -> np.ndarray`
- `intrinsic_from_paths(paths: np.ndarray) -> np.ndarray` (default == terminal_from_paths)
- `is_path_dependent -> bool` (True)
- `__call__(paths)` alias for `terminal_from_paths(paths)`

#### Paths array shape
- `paths.shape == (n_paths, n_steps + 1)`
- Column 0 is `S0`, last column is `S_T`

#### Where path payoffs are used
- **MC only** in V1/V1.1 (barriers, Asians, lookbacks, etc.)

### 4.3 Conventions
- Payoffs return **per-unit notional**. Pricers apply `trade.notional` and discounting.
- Payoffs are vectorized and return `float64` arrays.
- Use `_as_float_array` / `_as_paths_array` helpers for dtype + validation.

---

## 5) Pricer contract

Pricers live under `src/pricers/`.

### Minimal protocol (routing contract)

As implemented in `src/pricers/registry.py`:

```python
class InstrumentPricer(Protocol):
    def price(self, instrument: Any, market: Any) -> float:
        ...
```

Optional methods:
- `greeks(instrument, market) -> dict[str, float]`
- `diagnostics(instrument, market, ...) -> Any`

### Conventions
- `price(...)` returns PV in the instrument’s natural currency (e.g., domestic for FX options).
- Greeks are **scaled consistently with PV** (if PV includes notional, greeks include notional too).
- If greeks are not reliable (e.g., digitals without smoothing), pricer may return zeros or raise a documented error.
  Choose one policy and keep it consistent.

---

## 6) Registry and dispatch

Routing is done exclusively through `PricerRegistry`.

### V1 routing rules (already implemented)
Resolution order:
1) exact type match
2) walk `instrument_type.__mro__`
3) fallback `isinstance` scan (virtual subclasses)

Supports:
- default pricer per type (`pricer_id=None`)
- named pricers per type (`pricer_id="..."`) for model selection / benchmarking

### Default registry (V1)

Your current `DefaultPricerRegistry.build()` registers:

- `FxSpot -> FxSpotPricer`
- `FxForward -> FxForwardPricer`
- `EuropeanFxDigitalOption -> FxEuropeanDigitalBsmPricer`
- `EuropeanFxVanillaOption -> FxEuropeanVanillaBsmPricer`
- `EuropeanFxBarrierOption -> FxEuropeanBarrierMcPricer`
- `AmericanFxVanillaOption -> FxAmericanVanillaFdPricer`

**Policy note (recommended):**
- Barriers are **MC-only** in V1.1 (default to MC; do not provide FD unless explicitly intended and documented).

---

## 7) Model support matrix (V1 / V1.1)

This matrix is a **routing and test contract**. If a product-model cell is not supported, it must:
- not be registered, or
- be registered under a named `pricer_id` only, and documented as experimental.

### FX (current intent)
| Product | BSM | FD/PDE | MC |
|---|---:|---:|---:|
| European Vanilla | ✅ | ✅ | ✅ |
| European Digital | ✅ | ✅ (slower convergence) | ✅ |
| European Barrier | ❌ | ❌ (V1 policy) | ✅ |
| American Vanilla | ❌ | ✅ | (future) |

---

## 8) FD/PDE vs MC: “Does FD use the whole path?”

No.

- Standard 1D FD/PDE solves a backward PDE in state `(t, S)` (or `(t, log S)`), using only:
  - terminal payoff `V(T,S)`
  - boundary conditions / early exercise conditions (if American)
- MC uses full simulated paths and therefore uses `PathPayoff1D`.

**Continuous barriers** can be done with PDE using absorbing boundaries, but this is:
- easy to misinterpret,
- can diverge from discrete-monitoring MC,
- and increases maintenance cost.

Therefore, the default V1 policy is **barriers = MC-only**.

---

## 9) Adding a new product without refactors (checklist)

### Step A — Create instrument dataclass
- Put it under `src/instruments/<asset_class>/...`
- Include only trade fields + market ids

### Step B — Create payoff implementation
- If terminal-only: subclass `BasePayoff1D` and implement `terminal(spot)`
- If path-dependent: subclass `BasePathPayoff1D` and implement `terminal_from_paths(paths)`

### Step C — Implement pricer(s)
- Use the payoff library (no inline payoff logic inside pricer code)
- Keep pricer-specific model assumptions explicit (GK vs local vol, etc.)

### Step D — Register it
- Add to `DefaultPricerRegistry.build()` (default pricer) or register under a named `pricer_id`

### Step E — Tests (mandatory)
- Payoff unit tests (vectorization, edge cases)
- Cross-pricer parity tests when applicable:
  - BSM vs FD for vanillas
  - BSM vs MC (within MC error)
- If no parity possible, add a regression test with known value(s)

---

## 10) Test “alarm bell” suite (recommended)

These tests prevent accidental refactors:
- **Payoff conformance tests**
  - `terminal()` returns float64 ndarray, correct shape
  - `BasePathPayoff1D` rejects non-2D paths
- **Routing tests**
  - `registry.resolve()` returns expected pricer per instrument type
  - `pricer_id` selection behaves as expected
- **Parity tests**
  - Vanilla: BSM vs FD close
  - Vanilla: BSM vs MC within tolerance
  - Digital: looser tolerance

---

## 11) Versioning guide (V1 → Vn)

- **V1:** terminal-payoff products stable across BSM/FD/MC
- **V1.1:** introduce path payoffs + MC barriers (no FD barrier)
- **V2:** risk engine (bump & reprice) consumes pricers generically
- **V3:** calibration builders behind factories (Market interface unchanged)
- **Vn:** advanced numerics (AAD, local vol PDEs, discrete barrier PDE jump conditions, etc.)

---

## 12) “Do not do” list (prevents refactors)
- Do **not** duplicate payoff logic inside pricers.
- Do **not** make instruments depend on pricers.
- Do **not** change payoff signatures to “fit” a specific pricer.
- Do **not** add experimental support as default registration—use `pricer_id` and document it.

---

## Appendix: References in codebase
- Registry: `src/pricers/registry.py`
- Payoff base contracts: `src/models/payoffs/base.py`
