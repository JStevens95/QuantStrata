# LIBOR Market Model (LMM)

## Overview

The LIBOR Market Model (LMM), also known as the BGM model (Brace-Gatarek-Musiela), is the industry standard for pricing complex interest rate derivatives. Unlike short-rate models (Hull-White, Black-Karasinski), the LMM directly models observable forward LIBOR rates.

QuantStrata provides:
- **LMMCorrelation**: Correlation structures (flat, exponential, custom)
- **LMMParameters**: Model parameters including forwards, volatilities, correlation
- **LMMDynamics**: Monte Carlo simulation and derivative pricing

## Mathematical Framework

### Forward Rate Dynamics

Under the spot (rolling bank account) measure, each forward rate follows:

$$\frac{dF_i(t)}{F_i(t)} = \mu_i(t) \, dt + \sigma_i \, dW_i(t)$$

Where:
- $F_i(t)$: Forward LIBOR rate for period $[T_i, T_{i+1}]$
- $\sigma_i$: Instantaneous volatility
- $W_i(t)$: Brownian motion with correlation $\rho_{ij}$

### Drift Correction (No-Arbitrage)

Under the spot measure, the drift ensures no-arbitrage:

$$\mu_i(t) = \sum_{j=\beta(t)}^{i} \frac{\rho_{ij} \sigma_i \sigma_j \tau_j F_j(t)}{1 + \tau_j F_j(t)}$$

Where $\beta(t)$ is the index of the first forward not yet fixed at time $t$.

### Key Properties

1. **Log-normal forwards**: Each $F_i$ is always positive
2. **Correlated evolution**: Forward rates move together
3. **Exact caplet pricing**: Matches Black's formula at $t=0$
4. **No closed-form swaptions**: Requires Monte Carlo

## Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `tenors` | Forward fixing times $T_0, ..., T_n$ | 0, 0.25, 0.5, ..., 10Y |
| `accrual_factors` | Day count fractions $\tau_i$ | ~0.25 (quarterly) |
| `initial_forwards` | Initial rates $F_i(0)$ | 0.02 - 0.05 |
| `volatilities` | Instantaneous vols $\sigma_i$ | 0.15 - 0.25 |
| `correlation` | Correlation structure | See below |

### Correlation Types

| Type | Formula | Use Case |
|------|---------|----------|
| `flat` | $\rho_{ij} = \rho$ | Simple, testing |
| `exponential` | $\rho_{ij} = e^{-\beta|T_i - T_j|}$ | Most common |
| `custom` | User-defined matrix | Calibrated |

## Usage Examples

### Setting Up the Model

```python
import numpy as np
from src.models.forward_rate import (
    LMMCorrelation,
    LMMParameters,
    LMMDynamics,
)

# Define tenor structure (quarterly, 2 years)
n = 8
tenors = np.linspace(0.0, 2.0, n + 1)
accrual_factors = np.full(n, 0.25)

# Initial forward curve (upward sloping)
initial_forwards = 0.03 + 0.001 * np.arange(n)

# Volatility term structure (declining)
volatilities = 0.20 - 0.01 * np.arange(n)
volatilities = np.maximum(volatilities, 0.10)

# Exponential correlation
correlation = LMMCorrelation(
    n_forwards=n,
    correlation_type="exponential",
    beta=0.1,  # Decay parameter
)

# Create parameters
params = LMMParameters(
    tenors=tenors,
    accrual_factors=accrual_factors,
    initial_forwards=initial_forwards,
    volatilities=volatilities,
    correlation=correlation,
)

# Create dynamics
dynamics = LMMDynamics(params)

print(f"Number of forwards: {params.n_forwards}")
print(f"Terminal time: {params.terminal_time:.1f}Y")
```

### Monte Carlo Simulation

```python
# Simulate forward rate paths
sim = dynamics.simulate(
    n_paths=50000,
    n_steps_per_period=10,
    seed=42,
    antithetic=True,
)

print(f"Simulation shape: {sim.forwards.shape}")
# (n_paths, n_forwards, n_time_steps+1)

# Access forward rates at terminal time
F_terminal = sim.forwards[:, :, -1]
print(f"Mean terminal F_0: {F_terminal[:, 0].mean():.4f}")
```

### Pricing a Caplet

```python
# Price caplet on F_2 (fixing at T_2)
fixing_index = 2
strike = 0.034  # Close to ATM

caplet_price = dynamics.price_caplet(
    fixing_index=fixing_index,
    strike=strike,
    n_paths=100000,
    seed=42,
)
print(f"Caplet price: {caplet_price * 10000:.2f} bp (per unit notional)")
```

### Pricing a Floorlet

```python
floorlet_price = dynamics.price_floorlet(
    fixing_index=2,
    strike=0.034,
    n_paths=100000,
    seed=42,
)
print(f"Floorlet price: {floorlet_price * 10000:.2f} bp")

# Verify put-call parity
parity = (caplet_price - floorlet_price) * 10000
print(f"Caplet - Floorlet: {parity:.2f} bp")
```

### Pricing a Cap

```python
# Cap = sum of caplets
strike = 0.035
cap_price = 0.0

for i in range(n):
    caplet = dynamics.price_caplet(
        fixing_index=i,
        strike=strike,
        n_paths=50000,
        seed=42 + i,
    )
    cap_price += caplet

print(f"Cap price: {cap_price * 10000:.2f} bp (per unit notional)")
```

### Pricing a Swaption

```python
# 1Y into 1Y payer swaption
payer_swaption = dynamics.price_swaption(
    start_index=4,   # Swap starts at T_4 = 1Y
    end_index=8,     # Swap ends at T_8 = 2Y
    strike=0.035,
    is_payer=True,
    n_paths=100000,
    seed=42,
)
print(f"Payer swaption: {payer_swaption * 10000:.2f} bp")

# Receiver swaption
receiver_swaption = dynamics.price_swaption(
    start_index=4,
    end_index=8,
    strike=0.035,
    is_payer=False,
    n_paths=100000,
    seed=42,
)
print(f"Receiver swaption: {receiver_swaption * 10000:.2f} bp")
```

## Correlation Structures

### Flat Correlation

```python
# All pairs have same correlation
flat_corr = LMMCorrelation(
    n_forwards=8,
    correlation_type="flat",
    flat_corr=0.6,
)

matrix = flat_corr.get_correlation_matrix()
print(f"ρ_01 = {matrix[0, 1]:.2f}")
print(f"ρ_07 = {matrix[0, 7]:.2f}")  # Same
```

### Exponential Correlation

```python
# Nearby forwards more correlated
exp_corr = LMMCorrelation(
    n_forwards=8,
    correlation_type="exponential",
    beta=0.15,
)

matrix = exp_corr.get_correlation_matrix(tenors[:-1])
print(f"ρ_01 = {matrix[0, 1]:.3f}")  # High
print(f"ρ_07 = {matrix[0, 7]:.3f}")  # Lower
```

### Custom Correlation

```python
# User-provided matrix
custom_matrix = np.eye(4)
custom_matrix[0, 1] = custom_matrix[1, 0] = 0.8
custom_matrix[0, 2] = custom_matrix[2, 0] = 0.6
custom_matrix[0, 3] = custom_matrix[3, 0] = 0.4
custom_matrix[1, 2] = custom_matrix[2, 1] = 0.7
custom_matrix[1, 3] = custom_matrix[3, 1] = 0.5
custom_matrix[2, 3] = custom_matrix[3, 2] = 0.6

custom_corr = LMMCorrelation(
    n_forwards=4,
    correlation_type="custom",
    correlation_matrix=custom_matrix,
)
```

## Discretization

The model uses log-Euler discretization:

$$F_i(t+\Delta t) = F_i(t) \cdot \exp\left[(\mu_i - \sigma_i^2/2)\Delta t + \sigma_i \sqrt{\Delta t} \cdot Z_i\right]$$

Where $Z = L \cdot \xi$ with $L$ being the Cholesky factor and $\xi$ independent standard normals.

## When to Use LMM

### ✅ Use LMM For:

- **Caps and floors**: Standard pricing model
- **European swaptions**: Industry standard
- **Exotic IR derivatives**: Path-dependent payoffs
- **Correlation-sensitive products**: CMS, spread options
- **Smile calibration**: With extensions (SABR-LMM, stochastic vol)

### ❌ Don't Use LMM For:

- **Simple zero-coupon bonds**: Analytic formulas faster
- **Single caplet**: Black's formula is exact and faster
- **Negative rates**: Standard LMM has log-normal rates (use shifted LMM)

## Calibration

### To Caps/Caplets

LMM is exactly consistent with Black's caplet formula at $t=0$:
$$\text{Black Caplet} = \tau \cdot P(0, T_{i+1}) \cdot [F_i N(d_1) - K N(d_2)]$$

Where $\sigma_i$ is calibrated to match market cap vols.

### To Swaptions

Swaptions require numerical calibration since there's no closed-form solution. Common approaches:
1. **Rebonato approximation**: Analytic approximation for swaption vol
2. **Monte Carlo calibration**: Match simulated prices to market

### Correlation Calibration

Correlation affects:
- Swaption prices (more than caps)
- CMS convexity adjustments
- Correlation products

Calibrate using:
- Historical forward rate correlations
- Swaption smile/skew (indirectly)

## Interview Key Points

1. **Model Type**: Forward rate model (not short rate)
2. **Observables**: Models actual LIBOR forwards
3. **Log-Normal**: Rates always positive (no negative rates)
4. **Drift Correction**: Required for no-arbitrage under spot measure
5. **Correlation**: Critical for swaptions and exotics
6. **Caplet Consistency**: Exactly matches Black's formula
7. **No Closed-Form Swaptions**: MC or approximations needed

## Common Pitfalls

1. **Forgetting drift correction**: Violates no-arbitrage
2. **Wrong measure**: Drift depends on numeraire choice
3. **Correlation matrix not positive definite**: Cholesky fails
4. **Too few paths**: High variance in swaption prices
5. **Ignoring discretization bias**: Use enough time steps

## References

1. Brace, A., Gatarek, D., & Musiela, M. (1997). "The Market Model of Interest Rate Dynamics"
2. Rebonato, R. (2002). *Modern Pricing of Interest-Rate Derivatives*
3. Brigo, D. & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*
