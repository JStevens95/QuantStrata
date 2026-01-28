#!/usr/bin/env python3
"""
Equity Vanilla Options Pricing Example

This script demonstrates pricing European and American equity vanilla options
using multiple methods: BSM (analytic), Monte Carlo, and Finite Difference.

Key concepts covered:
1. Equity option modeling with continuous dividend yield
2. Cost-of-carry: b = r - q (vs FX where b = r_d - r_f)
3. Put-call parity for equity options
4. American vs European options (early exercise premium)
5. Comparison of BSM, MC, and FD pricing methods

Author: QuantStrata Team
"""

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

from src.instruments.equity.options.vanilla import (
    EuropeanEquityVanillaOption,
    AmericanEquityVanillaOption,
)
from src.pricers.equity.european_bsm import EquityEuropeanVanillaBsmPricer
from src.pricers.equity.european_mc import EquityEuropeanVanillaMcPricer
from src.pricers.equity.european_fde import EquityEuropeanVanillaFdPricer
from src.pricers.equity.american_fde import EquityAmericanVanillaFdPricer

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'font.size': 11,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
})

# =============================================================================
# Setup: Market and Instruments
# =============================================================================

print("=" * 70)
print("Equity Vanilla Options Pricing Example")
print("=" * 70)

# Define market IDs
AAPL_SPOT = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
AAPL_VOL = MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")
USD_CURVE = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")

# Market parameters
S0 = 150.0        # Spot price
r = 0.05          # Risk-free rate (5%)
q = 0.01          # Dividend yield (1%)
sigma = 0.25      # Volatility (25%)
K = 150.0         # Strike (ATM)
T = 1.0           # Time to expiry (1 year)

# Create market snapshot
market = Market(
    asof="2026-01-28",
    quotes={AAPL_SPOT: Quote(value=S0)},
    curves={USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r)},
    vols={AAPL_VOL: FlatVolSurface(sigma=sigma)},
)

print(f"\nMarket Setup:")
print(f"  Spot (S):           ${S0:.2f}")
print(f"  Strike (K):         ${K:.2f}")
print(f"  Expiry (T):         {T:.2f} years")
print(f"  Risk-free rate (r): {r*100:.1f}%")
print(f"  Dividend yield (q): {q*100:.1f}%")
print(f"  Volatility (σ):     {sigma*100:.1f}%")
print(f"  Cost-of-carry (b):  {(r-q)*100:.1f}% (r - q)")

# =============================================================================
# 1. European Vanilla: BSM, MC, FD Comparison
# =============================================================================

print("\n" + "=" * 70)
print("1. European Vanilla Pricing: Method Comparison")
print("=" * 70)

# Create European options
eu_call = EuropeanEquityVanillaOption(
    ticker="AAPL",
    option_type="call",
    strike=K,
    expiry=T,
    notional=1,
    dividend_yield=q,
    spot_id=AAPL_SPOT,
    vol_id=AAPL_VOL,
    curve_id=USD_CURVE,
)

eu_put = EuropeanEquityVanillaOption(
    ticker="AAPL",
    option_type="put",
    strike=K,
    expiry=T,
    notional=1,
    dividend_yield=q,
    spot_id=AAPL_SPOT,
    vol_id=AAPL_VOL,
    curve_id=USD_CURVE,
)

# Create pricers
bsm_pricer = EquityEuropeanVanillaBsmPricer()
mc_pricer = EquityEuropeanVanillaMcPricer(n_paths=500_000, seed=42, antithetic=True)
fd_pricer = EquityEuropeanVanillaFdPricer(n_space=401, n_time_steps=200)

# Price with all methods
bsm_call = bsm_pricer.price(eu_call, market)
bsm_put = bsm_pricer.price(eu_put, market)

mc_call = mc_pricer.price(eu_call, market)
mc_put = mc_pricer.price(eu_put, market)

fd_call = fd_pricer.price(eu_call, market)
fd_put = fd_pricer.price(eu_put, market)

print(f"\nEuropean Option Prices:")
print(f"{'Method':<15} {'Call':<12} {'Put':<12}")
print(f"{'-'*39}")
print(f"{'BSM (Analytic)':<15} ${bsm_call:<11.4f} ${bsm_put:<11.4f}")
print(f"{'Monte Carlo':<15} ${mc_call:<11.4f} ${mc_put:<11.4f}")
print(f"{'Finite Diff':<15} ${fd_call:<11.4f} ${fd_put:<11.4f}")

# Verify convergence
print(f"\nConvergence to BSM:")
print(f"  MC Call Error:  {abs(mc_call - bsm_call):.6f} ({100*abs(mc_call - bsm_call)/bsm_call:.3f}%)")
print(f"  MC Put Error:   {abs(mc_put - bsm_put):.6f} ({100*abs(mc_put - bsm_put)/bsm_put:.3f}%)")
print(f"  FD Call Error:  {abs(fd_call - bsm_call):.6f} ({100*abs(fd_call - bsm_call)/bsm_call:.3f}%)")
print(f"  FD Put Error:   {abs(fd_put - bsm_put):.6f} ({100*abs(fd_put - bsm_put)/bsm_put:.3f}%)")

# =============================================================================
# 2. Put-Call Parity Verification
# =============================================================================

print("\n" + "=" * 70)
print("2. Put-Call Parity Verification")
print("=" * 70)

# Put-Call Parity for equity: C - P = S*exp(-qT) - K*exp(-rT)
import math
parity_rhs = S0 * math.exp(-q * T) - K * math.exp(-r * T)
parity_lhs = bsm_call - bsm_put

print(f"\nPut-Call Parity: C - P = S*exp(-qT) - K*exp(-rT)")
print(f"  LHS (C - P):                 ${parity_lhs:.6f}")
print(f"  RHS (S*exp(-qT) - K*exp(-rT)): ${parity_rhs:.6f}")
print(f"  Difference:                    ${abs(parity_lhs - parity_rhs):.10f}")
print(f"  ✓ Parity holds!" if abs(parity_lhs - parity_rhs) < 1e-8 else "  ✗ Parity violated!")

# =============================================================================
# 3. Greeks Analysis
# =============================================================================

print("\n" + "=" * 70)
print("3. Greeks Analysis")
print("=" * 70)

greeks_call = bsm_pricer.greeks(eu_call, market)
greeks_put = bsm_pricer.greeks(eu_put, market)

print(f"\nBSM Greeks (per share):")
print(f"{'Greek':<10} {'Call':<12} {'Put':<12} {'Relationship'}")
print(f"{'-'*50}")
print(f"{'Delta':<10} {greeks_call['delta']:<12.4f} {greeks_put['delta']:<12.4f} Δ_put = Δ_call - exp(-qT)")
print(f"{'Gamma':<10} {greeks_call['gamma']:<12.4f} {greeks_put['gamma']:<12.4f} Same for call/put")
print(f"{'Vega':<10} {greeks_call['vega']:<12.4f} {greeks_put['vega']:<12.4f} Same for call/put")
print(f"{'Rho':<10} {greeks_call['rho']:<12.4f} {greeks_put['rho']:<12.4f} Opposite signs")

# Verify delta relationship
delta_diff = greeks_put['delta'] - (greeks_call['delta'] - math.exp(-q * T))
print(f"\nDelta Parity Check: Δ_put - (Δ_call - exp(-qT)) = {delta_diff:.8f}")

# =============================================================================
# 4. American vs European Comparison
# =============================================================================

print("\n" + "=" * 70)
print("4. American vs European: Early Exercise Premium")
print("=" * 70)

# American pricers
am_fd_pricer = EquityAmericanVanillaFdPricer(n_space=401, n_time_steps=200)

# Create American options
am_call = AmericanEquityVanillaOption(
    ticker="AAPL",
    option_type="call",
    strike=K,
    expiry=T,
    notional=1,
    dividend_yield=q,
    spot_id=AAPL_SPOT,
    vol_id=AAPL_VOL,
    curve_id=USD_CURVE,
)

am_put = AmericanEquityVanillaOption(
    ticker="AAPL",
    option_type="put",
    strike=K,
    expiry=T,
    notional=1,
    dividend_yield=q,
    spot_id=AAPL_SPOT,
    vol_id=AAPL_VOL,
    curve_id=USD_CURVE,
)

# Price American options
am_call_pv = am_fd_pricer.price(am_call, market)
am_put_pv = am_fd_pricer.price(am_put, market)

print(f"\nAmerican vs European Prices:")
print(f"{'Option':<15} {'European':<12} {'American':<12} {'Premium':<12}")
print(f"{'-'*51}")
print(f"{'Call':<15} ${bsm_call:<11.4f} ${am_call_pv:<11.4f} ${am_call_pv - bsm_call:<11.4f}")
print(f"{'Put':<15} ${bsm_put:<11.4f} ${am_put_pv:<11.4f} ${am_put_pv - bsm_put:<11.4f}")

print(f"\nKey Insights:")
print(f"  - American Call Premium: ${am_call_pv - bsm_call:.4f}")
print(f"    (Small because dividend yield q={q*100:.1f}% is low)")
print(f"  - American Put Premium:  ${am_put_pv - bsm_put:.4f}")
print(f"    (Puts always have early exercise value when ITM)")

# =============================================================================
# 5. Dividend Impact Analysis
# =============================================================================

print("\n" + "=" * 70)
print("5. Dividend Impact Analysis")
print("=" * 70)

dividend_yields = np.linspace(0, 0.10, 11)  # 0% to 10%
eu_call_prices = []
eu_put_prices = []
am_call_prices = []
am_put_prices = []

for q_val in dividend_yields:
    # Create option with this dividend yield
    eu_c = EuropeanEquityVanillaOption(
        ticker="AAPL", option_type="call", strike=K, expiry=T,
        notional=1, dividend_yield=q_val, spot_id=AAPL_SPOT,
        vol_id=AAPL_VOL, curve_id=USD_CURVE,
    )
    eu_p = EuropeanEquityVanillaOption(
        ticker="AAPL", option_type="put", strike=K, expiry=T,
        notional=1, dividend_yield=q_val, spot_id=AAPL_SPOT,
        vol_id=AAPL_VOL, curve_id=USD_CURVE,
    )
    am_c = AmericanEquityVanillaOption(
        ticker="AAPL", option_type="call", strike=K, expiry=T,
        notional=1, dividend_yield=q_val, spot_id=AAPL_SPOT,
        vol_id=AAPL_VOL, curve_id=USD_CURVE,
    )
    am_p = AmericanEquityVanillaOption(
        ticker="AAPL", option_type="put", strike=K, expiry=T,
        notional=1, dividend_yield=q_val, spot_id=AAPL_SPOT,
        vol_id=AAPL_VOL, curve_id=USD_CURVE,
    )
    
    eu_call_prices.append(bsm_pricer.price(eu_c, market))
    eu_put_prices.append(bsm_pricer.price(eu_p, market))
    am_call_prices.append(am_fd_pricer.price(am_c, market))
    am_put_prices.append(am_fd_pricer.price(am_p, market))

print(f"\nDividend Yield Impact on Option Prices:")
print(f"{'q':<8} {'EU Call':<10} {'AM Call':<10} {'EU Put':<10} {'AM Put':<10}")
print(f"{'-'*48}")
for i, q_val in enumerate(dividend_yields[::2]):  # Every other value
    idx = i * 2
    print(f"{q_val*100:.1f}%    ${eu_call_prices[idx]:<9.4f} ${am_call_prices[idx]:<9.4f} ${eu_put_prices[idx]:<9.4f} ${am_put_prices[idx]:<9.4f}")

# =============================================================================
# Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Option prices vs spot
spots = np.linspace(100, 200, 50)
call_prices_vs_spot = []
put_prices_vs_spot = []

for s in spots:
    mkt = Market(
        asof="2026-01-28",
        quotes={AAPL_SPOT: Quote(value=s)},
        curves={USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r)},
        vols={AAPL_VOL: FlatVolSurface(sigma=sigma)},
    )
    call_prices_vs_spot.append(bsm_pricer.price(eu_call, mkt))
    put_prices_vs_spot.append(bsm_pricer.price(eu_put, mkt))

ax1 = axes[0, 0]
ax1.plot(spots, call_prices_vs_spot, 'b-', label='Call')
ax1.plot(spots, put_prices_vs_spot, 'r-', label='Put')
ax1.axvline(x=K, color='gray', linestyle='--', alpha=0.5, label=f'Strike = ${K}')
ax1.set_xlabel('Spot Price ($)')
ax1.set_ylabel('Option Price ($)')
ax1.set_title('Option Price vs Spot')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: American vs European premium
ax2 = axes[0, 1]
am_call_premium = np.array(am_call_prices) - np.array(eu_call_prices)
am_put_premium = np.array(am_put_prices) - np.array(eu_put_prices)
ax2.plot(dividend_yields * 100, am_call_premium, 'b-', label='Call Premium')
ax2.plot(dividend_yields * 100, am_put_premium, 'r-', label='Put Premium')
ax2.set_xlabel('Dividend Yield (%)')
ax2.set_ylabel('Early Exercise Premium ($)')
ax2.set_title('American - European Premium vs Dividend Yield')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Delta vs spot
call_deltas = []
put_deltas = []
for s in spots:
    mkt = Market(
        asof="2026-01-28",
        quotes={AAPL_SPOT: Quote(value=s)},
        curves={USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r)},
        vols={AAPL_VOL: FlatVolSurface(sigma=sigma)},
    )
    call_deltas.append(bsm_pricer.greeks(eu_call, mkt)['delta'])
    put_deltas.append(bsm_pricer.greeks(eu_put, mkt)['delta'])

ax3 = axes[1, 0]
ax3.plot(spots, call_deltas, 'b-', label='Call Delta')
ax3.plot(spots, put_deltas, 'r-', label='Put Delta')
ax3.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax3.set_xlabel('Spot Price ($)')
ax3.set_ylabel('Delta')
ax3.set_title('Delta vs Spot')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Gamma vs spot
call_gammas = []
for s in spots:
    mkt = Market(
        asof="2026-01-28",
        quotes={AAPL_SPOT: Quote(value=s)},
        curves={USD_CURVE: FlatZeroRateCurve(continuously_compounded_rate=r)},
        vols={AAPL_VOL: FlatVolSurface(sigma=sigma)},
    )
    call_gammas.append(bsm_pricer.greeks(eu_call, mkt)['gamma'])

ax4 = axes[1, 1]
ax4.plot(spots, call_gammas, 'g-', label='Gamma')
ax4.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Spot Price ($)')
ax4.set_ylabel('Gamma')
ax4.set_title('Gamma vs Spot (Peak at ATM)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('equity_vanilla_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("Analysis complete. Plot saved to equity_vanilla_analysis.png")
print("=" * 70)
