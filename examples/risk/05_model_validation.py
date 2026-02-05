#!/usr/bin/env python3
"""
===============================================================================
Risk: Model Validation and Stress Testing
===============================================================================

This example demonstrates comprehensive model validation techniques used in
production risk systems - comparing models, stress testing, and quantifying
model uncertainty.

Learning Objectives
-------------------
1. **Cross-Model Validation**: Compare BSM vs Monte Carlo vs FDE
2. **Convergence Testing**: Verify numerical methods converge properly
3. **Extreme Scenario Stress**: Test models under extreme conditions
4. **Model Risk Quantification**: Understand limits of each approach

Mathematical Framework
----------------------
Model validation involves:

1. Consistency: Different methods converge to same value
   |V_BSM - V_MC| < ε for large N simulations
   |V_BSM - V_FDE| < ε for fine grids

2. Convergence: Numerical methods converge at expected rates
   Error_MC ∝ 1/√N    (Monte Carlo)
   Error_FDE ∝ Δx²    (Finite difference, 2nd order)

3. Boundary Behavior: Correct limits
   V(S→∞, Call) → S - K·e^(-rT)
   V(S→0, Call) → 0

Production Context
------------------
At a hedge fund:
- Model validation is required by risk management
- Regulators demand documented model testing
- P&L attribution reveals model weaknesses
- Stress tests expose hidden risks

Prerequisites
-------------
- Basic option pricing
- Numerical methods understanding
- Previous risk examples

Run This Example
----------------
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/risk/05_model_validation.py

Author: QuantStrata Team
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

ENABLE_PLOTTING = True

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - plotting disabled")


# =============================================================================
# ANALYTICAL MODEL (BSM)
# =============================================================================

def bs_d1d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """Calculate BSM d1 and d2."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0)
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> float:
    """Black-Scholes delta."""
    if T <= 0:
        return 1.0 if (S > K and is_call) else (0.0 if is_call else -1.0 if S < K else 0.0)
    d1, _ = bs_d1d2(S, K, T, r, sigma)
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma."""
    if T <= 0:
        return 0.0
    d1, _ = bs_d1d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega."""
    if T <= 0:
        return 0.0
    d1, _ = bs_d1d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change


# =============================================================================
# MONTE CARLO MODEL
# =============================================================================

def mc_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Monte Carlo call price with standard error.
    
    Returns
    -------
    Tuple
        (price, standard_error)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # GBM simulation
    Z = np.random.randn(n_paths)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Discounted payoffs
    payoffs = np.exp(-r * T) * np.maximum(S_T - K, 0)
    
    price = np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error


def mc_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Monte Carlo put price with standard error."""
    if seed is not None:
        np.random.seed(seed)
    
    Z = np.random.randn(n_paths)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    payoffs = np.exp(-r * T) * np.maximum(K - S_T, 0)
    
    price = np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error


# =============================================================================
# FINITE DIFFERENCE MODEL
# =============================================================================

def fde_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_space: int = 200,
    n_time: int = 100,
) -> float:
    """
    Finite difference call price using Crank-Nicolson scheme.
    """
    # Grid setup
    S_max = 4 * K  # Upper boundary
    dS = S_max / n_space
    dt = T / n_time
    
    S_grid = np.linspace(0, S_max, n_space + 1)
    
    # Terminal condition
    V = np.maximum(S_grid - K, 0)
    
    # Coefficients for tridiagonal system
    i = np.arange(1, n_space)
    
    # Crank-Nicolson coefficients
    alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
    beta = -0.5 * dt * (sigma**2 * i**2 + r)
    gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)
    
    # Build matrices
    M1 = np.diag(1 - beta) + np.diag(-alpha[1:], -1) + np.diag(-gamma[:-1], 1)
    M2 = np.diag(1 + beta) + np.diag(alpha[1:], -1) + np.diag(gamma[:-1], 1)
    
    # Time stepping
    for _ in range(n_time):
        # Boundary conditions
        V[0] = 0  # Call at S=0
        V[-1] = S_max - K * np.exp(-r * (_ + 1) * dt)  # Call at S_max
        
        # Interior points
        rhs = M2 @ V[1:-1]
        rhs[0] += alpha[0] * (V[0])
        rhs[-1] += gamma[-1] * (V[-1])
        
        V[1:-1] = np.linalg.solve(M1, rhs)
    
    # Interpolate to get price at S
    idx = int(S / dS)
    idx = min(idx, n_space - 1)
    weight = (S - S_grid[idx]) / dS
    
    return V[idx] * (1 - weight) + V[idx + 1] * weight


# =============================================================================
# VALIDATION TESTS
# =============================================================================

@dataclass
class ValidationResult:
    """Container for validation test results."""
    test_name: str
    passed: bool
    details: dict
    message: str


def test_cross_model_consistency(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 0.5,
    r: float = 0.05,
    sigma: float = 0.2,
    tol: float = 0.01,
) -> ValidationResult:
    """
    Test that BSM, MC, and FDE produce consistent prices.
    """
    # BSM (benchmark)
    bsm_price = bs_call_price(S, K, T, r, sigma)
    
    # Monte Carlo
    mc_price, mc_se = mc_call_price(S, K, T, r, sigma, n_paths=500000, seed=42)
    
    # Finite Difference
    fde_price = fde_call_price(S, K, T, r, sigma, n_space=400, n_time=200)
    
    # Calculate deviations
    mc_error = abs(mc_price - bsm_price) / bsm_price
    fde_error = abs(fde_price - bsm_price) / bsm_price
    
    passed = mc_error < tol and fde_error < tol
    
    return ValidationResult(
        test_name="Cross-Model Consistency",
        passed=passed,
        details={
            'bsm_price': bsm_price,
            'mc_price': mc_price,
            'mc_se': mc_se,
            'fde_price': fde_price,
            'mc_error_pct': mc_error * 100,
            'fde_error_pct': fde_error * 100,
        },
        message=f"BSM={bsm_price:.4f}, MC={mc_price:.4f}±{mc_se:.4f}, FDE={fde_price:.4f}"
    )


def test_mc_convergence(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 0.5,
    r: float = 0.05,
    sigma: float = 0.2,
) -> ValidationResult:
    """
    Test Monte Carlo convergence rate (should be O(1/√N)).
    """
    bsm_price = bs_call_price(S, K, T, r, sigma)
    
    path_counts = [1000, 5000, 10000, 50000, 100000, 500000]
    errors = []
    std_errors = []
    
    for n in path_counts:
        mc_price, mc_se = mc_call_price(S, K, T, r, sigma, n_paths=n, seed=42)
        errors.append(abs(mc_price - bsm_price))
        std_errors.append(mc_se)
    
    # Theoretical: error ∝ 1/√N
    # Check if error ratio follows expected pattern
    ratios = [errors[i] / errors[i-1] for i in range(1, len(errors))]
    expected_ratios = [np.sqrt(path_counts[i-1] / path_counts[i]) for i in range(1, len(path_counts))]
    
    # Convergence is reasonable if ratios are within factor of 2
    ratio_errors = [abs(r - e) / e for r, e in zip(ratios, expected_ratios)]
    passed = all(r < 1.0 for r in ratio_errors)  # Within 100% of expected
    
    return ValidationResult(
        test_name="MC Convergence",
        passed=passed,
        details={
            'path_counts': path_counts,
            'errors': errors,
            'std_errors': std_errors,
            'ratios': ratios,
            'expected_ratios': expected_ratios,
        },
        message=f"Final error: {errors[-1]:.6f} at N={path_counts[-1]:,}"
    )


def test_put_call_parity(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 0.5,
    r: float = 0.05,
    sigma: float = 0.2,
    tol: float = 1e-6,
) -> ValidationResult:
    """
    Test put-call parity: C - P = S - K*e^(-rT)
    """
    call = bs_call_price(S, K, T, r, sigma)
    put = bs_put_price(S, K, T, r, sigma)
    
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    error = abs(lhs - rhs)
    
    passed = error < tol
    
    return ValidationResult(
        test_name="Put-Call Parity",
        passed=passed,
        details={
            'call': call,
            'put': put,
            'lhs': lhs,
            'rhs': rhs,
            'error': error,
        },
        message=f"C-P={lhs:.6f}, S-Ke^(-rT)={rhs:.6f}, Error={error:.2e}"
    )


def test_boundary_conditions(
    K: float = 100.0,
    T: float = 0.5,
    r: float = 0.05,
    sigma: float = 0.2,
    tol: float = 0.01,
) -> ValidationResult:
    """
    Test option boundary conditions.
    """
    results = {}
    
    # Call at S→0 should be 0
    call_s0 = bs_call_price(0.01, K, T, r, sigma)
    results['call_s0'] = call_s0
    
    # Put at S→∞ should be 0
    put_sinf = bs_put_price(10 * K, K, T, r, sigma)
    results['put_sinf'] = put_sinf
    
    # Call at S→∞ should be ≈ S - K*e^(-rT)
    S_high = 10 * K
    call_sinf = bs_call_price(S_high, K, T, r, sigma)
    call_sinf_expected = S_high - K * np.exp(-r * T)
    call_sinf_error = abs(call_sinf - call_sinf_expected) / call_sinf_expected
    results['call_sinf'] = call_sinf
    results['call_sinf_expected'] = call_sinf_expected
    
    # At expiry, intrinsic value
    call_expiry = bs_call_price(110, K, 0.0001, r, sigma)
    results['call_expiry'] = call_expiry
    results['call_expiry_expected'] = 10.0
    
    passed = (
        call_s0 < tol and
        put_sinf < tol and
        call_sinf_error < tol
    )
    
    return ValidationResult(
        test_name="Boundary Conditions",
        passed=passed,
        details=results,
        message=f"Call(S≈0)={call_s0:.6f}, Put(S→∞)={put_sinf:.6f}"
    )


def test_greeks_consistency(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 0.5,
    r: float = 0.05,
    sigma: float = 0.2,
    tol: float = 0.01,
) -> ValidationResult:
    """
    Test Greeks via finite difference vs analytical.
    """
    # Analytical Greeks
    delta_analytical = bs_delta(S, K, T, r, sigma)
    gamma_analytical = bs_gamma(S, K, T, r, sigma)
    
    # Numerical Greeks
    dS = 0.01 * S
    V_up = bs_call_price(S + dS, K, T, r, sigma)
    V_mid = bs_call_price(S, K, T, r, sigma)
    V_down = bs_call_price(S - dS, K, T, r, sigma)
    
    delta_numerical = (V_up - V_down) / (2 * dS)
    gamma_numerical = (V_up - 2 * V_mid + V_down) / (dS ** 2)
    
    delta_error = abs(delta_analytical - delta_numerical) / delta_analytical
    gamma_error = abs(gamma_analytical - gamma_numerical) / gamma_analytical
    
    passed = delta_error < tol and gamma_error < tol
    
    return ValidationResult(
        test_name="Greeks Consistency",
        passed=passed,
        details={
            'delta_analytical': delta_analytical,
            'delta_numerical': delta_numerical,
            'delta_error_pct': delta_error * 100,
            'gamma_analytical': gamma_analytical,
            'gamma_numerical': gamma_numerical,
            'gamma_error_pct': gamma_error * 100,
        },
        message=f"Delta error: {delta_error*100:.4f}%, Gamma error: {gamma_error*100:.4f}%"
    )


# =============================================================================
# STRESS TESTING
# =============================================================================

@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    spot_shock: float  # Multiplicative
    vol_shock: float  # Additive (percentage points)
    rate_shock: float  # Additive (basis points / 100)


def run_stress_tests(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 0.5,
    r: float = 0.05,
    sigma: float = 0.2,
) -> Dict[str, dict]:
    """
    Run model through stress scenarios.
    """
    scenarios = [
        StressScenario("Base Case", 1.0, 0.0, 0.0),
        StressScenario("Market Crash (-20%)", 0.8, 0.15, -0.01),
        StressScenario("Flash Crash (-40%)", 0.6, 0.30, -0.02),
        StressScenario("Bull Market (+30%)", 1.3, -0.05, 0.01),
        StressScenario("Vol Spike", 1.0, 0.20, 0.0),
        StressScenario("Vol Collapse", 1.0, -0.10, 0.0),
        StressScenario("Rate Hike", 1.0, 0.0, 0.02),
        StressScenario("Rate Cut", 1.0, 0.0, -0.02),
        StressScenario("Extreme: 2008 Crisis", 0.5, 0.40, -0.03),
        StressScenario("Extreme: Vol Explosion", 1.0, 0.50, 0.0),
    ]
    
    results = {}
    
    for scenario in scenarios:
        S_stressed = S * scenario.spot_shock
        sigma_stressed = max(0.01, sigma + scenario.vol_shock)
        r_stressed = r + scenario.rate_shock
        
        call_price = bs_call_price(S_stressed, K, T, r_stressed, sigma_stressed)
        put_price = bs_put_price(S_stressed, K, T, r_stressed, sigma_stressed)
        delta = bs_delta(S_stressed, K, T, r_stressed, sigma_stressed)
        gamma = bs_gamma(S_stressed, K, T, r_stressed, sigma_stressed)
        vega = bs_vega(S_stressed, K, T, r_stressed, sigma_stressed)
        
        results[scenario.name] = {
            'scenario': scenario,
            'call': call_price,
            'put': put_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'spot': S_stressed,
            'vol': sigma_stressed,
            'rate': r_stressed,
        }
    
    return results


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_model_validation() -> Tuple[List[ValidationResult], Dict[str, dict]]:
    """
    Run complete model validation suite.
    
    Returns
    -------
    Tuple
        List of validation results and stress test results.
    """
    logger.info("=" * 70)
    logger.info("SECTION 1: Model Consistency Tests")
    logger.info("=" * 70)
    
    validation_results = []
    
    # Run validation tests
    tests = [
        test_cross_model_consistency,
        test_mc_convergence,
        test_put_call_parity,
        test_boundary_conditions,
        test_greeks_consistency,
    ]
    
    for test_func in tests:
        logger.info("")
        result = test_func()
        validation_results.append(result)
        
        status = "PASS" if result.passed else "FAIL"
        logger.info(f"  [{status}] {result.test_name}")
        logger.info(f"         {result.message}")
    
    # Summary
    passed = sum(1 for r in validation_results if r.passed)
    total = len(validation_results)
    
    logger.info("")
    logger.info("-" * 70)
    logger.info(f"  Validation Summary: {passed}/{total} tests passed")
    logger.info("-" * 70)
    
    # Stress tests
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 2: Stress Testing")
    logger.info("=" * 70)
    
    stress_results = run_stress_tests()
    
    logger.info("")
    logger.info(f"{'Scenario':<25} {'Call':>10} {'Put':>10} {'Delta':>8} {'Gamma':>8}")
    logger.info("-" * 70)
    
    base_call = stress_results['Base Case']['call']
    
    for name, data in stress_results.items():
        pnl_pct = (data['call'] - base_call) / base_call * 100 if base_call > 0 else 0
        logger.info(
            f"{name:<25} {data['call']:>10.4f} {data['put']:>10.4f} "
            f"{data['delta']:>8.4f} {data['gamma']:>8.4f}"
        )
    
    return validation_results, stress_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_validation(
    validation_results: List[ValidationResult],
    stress_results: Dict[str, dict],
) -> None:
    """Visualize validation and stress test results."""
    if not MATPLOTLIB_AVAILABLE or not ENABLE_PLOTTING:
        logger.info("Skipping plots (matplotlib not available or disabled)")
        return
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SECTION 3: Visualization")
    logger.info("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # -------------------------------------------------------------------------
    # Plot 1: MC Convergence
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    
    mc_result = next((r for r in validation_results if r.test_name == "MC Convergence"), None)
    
    if mc_result:
        path_counts = mc_result.details['path_counts']
        errors = mc_result.details['errors']
        
        ax.loglog(path_counts, errors, 'o-', color='#2E86AB', linewidth=2, markersize=8)
        
        # Theoretical line
        ref_error = errors[0]
        ref_n = path_counts[0]
        theoretical = [ref_error * np.sqrt(ref_n / n) for n in path_counts]
        ax.loglog(path_counts, theoretical, '--', color='#E94F37', linewidth=2, label='O(1/√N)')
        
        ax.set_xlabel('Number of Paths')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Monte Carlo Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Model Comparison by Strike
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    
    S = 100.0
    T = 0.5
    r = 0.05
    sigma = 0.2
    
    strikes = np.linspace(70, 130, 13)
    bsm_prices = [bs_call_price(S, K, T, r, sigma) for K in strikes]
    mc_prices = [mc_call_price(S, K, T, r, sigma, n_paths=50000, seed=42)[0] for K in strikes]
    fde_prices = [fde_call_price(S, K, T, r, sigma, n_space=200, n_time=100) for K in strikes]
    
    ax.plot(strikes, bsm_prices, '-', color='#2E86AB', linewidth=2, label='BSM')
    ax.plot(strikes, mc_prices, 's', color='#E94F37', markersize=8, label='Monte Carlo')
    ax.plot(strikes, fde_prices, '^', color='#4CAF50', markersize=8, label='FDE')
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('Call Price')
    ax.set_title('Cross-Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Stress Test Results
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    
    scenarios = list(stress_results.keys())
    call_prices = [stress_results[s]['call'] for s in scenarios]
    
    colors = ['#2E86AB' if s == 'Base Case' else '#E94F37' if 'Crash' in s or 'Crisis' in s else '#4CAF50' for s in scenarios]
    
    bars = ax.barh(scenarios, call_prices, color=colors, alpha=0.8)
    ax.axvline(x=stress_results['Base Case']['call'], color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Call Price')
    ax.set_title('Stress Test: Call Prices by Scenario')
    ax.grid(True, alpha=0.3, axis='x')
    
    # -------------------------------------------------------------------------
    # Plot 4: Greeks Under Stress
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    deltas = [stress_results[s]['delta'] for s in scenarios]
    gammas = [stress_results[s]['gamma'] * 100 for s in scenarios]  # Scale for visibility
    
    ax.bar(x - width/2, deltas, width, label='Delta', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, gammas, width, label='Gamma (×100)', color='#E94F37', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('Greek Value')
    ax.set_title('Greeks Under Stress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    logger.info("Visualization complete")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary() -> None:
    """Print summary of key concepts."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    summary = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         KEY TAKEAWAYS                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. Model Consistency:                                              │
    │     - BSM, MC, FDE should agree (within tolerances)                │
    │     - Discrepancies indicate implementation bugs                    │
    │     - Regular validation catches drift                              │
    │                                                                      │
    │  2. Convergence Testing:                                            │
    │     - MC: error ∝ 1/√N (halve error = 4× paths)                    │
    │     - FDE: error ∝ Δx² (halve error = 4× grid points)              │
    │     - Verify expected convergence rates                             │
    │                                                                      │
    │  3. Stress Testing:                                                 │
    │     - Test extreme market conditions                                │
    │     - Identify model breakdown points                               │
    │     - Key for risk management                                       │
    │                                                                      │
    │  4. Production Best Practices:                                      │
    │     - Automated daily validation                                    │
    │     - Version control for model changes                             │
    │     - Document all test results                                     │
    │     - Alert on test failures                                        │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    logger.info(summary)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the example.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    global ENABLE_PLOTTING
    ENABLE_PLOTTING = args.plot
    
    try:
        # Run validation suite
        validation_results, stress_results = run_model_validation()
        
        # Visualization
        visualize_validation(validation_results, stress_results)
        
        # Summary
        print_summary()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model Validation Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Enable plotting (default: True)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable plotting",
    )
    
    args = parser.parse_args()
    main(args)
