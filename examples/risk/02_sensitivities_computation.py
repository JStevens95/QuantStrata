#!/usr/bin/env python3
"""
Sensitivities Computation: Greeks and Risk Factors

This example demonstrates how to:
- Compute Greeks analytically and via bump-and-reprice
- Use the sensitivities engine for comprehensive risk
- Compare analytical vs finite difference Greeks
- Generate sensitivity reports

Author: QuantStrata Team
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.marketdata.scenarios.shocks import SpotShock, VolShock, ParallelRateShock

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import PricerRegistry
from src.pricers.fx.european_bsm import EuropeanFxVanillaBsmPricer

from src.risk.sensitivities.engine import compute_sensitivities
from src.risk.sensitivities.config import SensitivitiesConfig, BumpConfig

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'lines.linewidth': 2,
})

# =============================================================================
# Setup
# =============================================================================

print("=" * 70)
print("Setup: Market and Portfolio")
print("=" * 70)

# Market IDs
EURUSD_SPOT = MarketId(asset_class="FX", data_type="SPOT", name="EURUSD")
USD_CURVE = MarketId(asset_class="IR", data_type="CURVE", name="USD_OIS")
EUR_CURVE = MarketId(asset_class="IR", data_type="CURVE", name="EUR_OIS")
EURUSD_VOL = MarketId(asset_class="FX", data_type="VOL", name="EURUSD")

# Market
spot = 1.0850
r_usd = 0.05
r_eur = 0.02
vol = 0.10

market = Market(
    asof="2026-01-28",
    quotes={EURUSD_SPOT: Quote(value=spot)},
    curves={USD_CURVE: FlatCurve(rate=r_usd), EUR_CURVE: FlatCurve(rate=r_eur)},
    vols={EURUSD_VOL: FlatVolSurface(vol=vol)},
)

# Option and portfolio
option = EuropeanFxVanillaOption(
    ccy_pair="EURUSD", strike=1.10, expiry_years=1.0, is_call=True,
    notional=10_000_000, spot_id=EURUSD_SPOT, domestic_curve_id=USD_CURVE,
    foreign_curve_id=EUR_CURVE, vol_id=EURUSD_VOL,
)

portfolio = Portfolio(positions=[
    Position(position_id="OTM_CALL", instrument=option, quantity=1.0)
])

# Pricer
registry = PricerRegistry()
registry.register(EuropeanFxVanillaOption, EuropeanFxVanillaBsmPricer())
portfolio_pricer = PortfolioPricer(pricer_registry=registry)
bsm_pricer = EuropeanFxVanillaBsmPricer()

# Base results
base_result = portfolio_pricer.price(portfolio, market)
base_pv = base_result.totals.pv
analytic_greeks = bsm_pricer.greeks(option, market)

print(f"\nOption: OTM EUR Call, K=1.10, T=1Y, Notional=10M EUR")
print(f"  Spot: {spot}, Vol: {vol:.1%}")
print(f"  Base PV: {base_pv:,.2f} USD")

print(f"\nAnalytical Greeks (BSM):")
for greek, value in analytic_greeks.items():
    print(f"  {greek}: {value:,.4f}")

# =============================================================================
# 1. Manual Bump-and-Reprice
# =============================================================================

print("\n" + "=" * 70)
print("1. Manual Bump-and-Reprice Greeks")
print("=" * 70)

def bump_and_reprice_delta(portfolio, market, pricer, spot_id, bump_pct=0.01):
    """Compute delta via central difference."""
    spot = market.quote(spot_id)
    h = spot * bump_pct
    
    # Bump up
    shock_up = SpotShock(name="up", spot_id=spot_id, bump=bump_pct, bump_mode="relative")
    market_up = shock_up.apply(market)
    pv_up = pricer.price(portfolio, market_up).totals.pv
    
    # Bump down
    shock_down = SpotShock(name="down", spot_id=spot_id, bump=-bump_pct, bump_mode="relative")
    market_down = shock_down.apply(market)
    pv_down = pricer.price(portfolio, market_down).totals.pv
    
    delta = (pv_up - pv_down) / (2 * h)
    return delta

def bump_and_reprice_gamma(portfolio, market, pricer, spot_id, bump_pct=0.01):
    """Compute gamma via central difference."""
    spot = market.quote(spot_id)
    h = spot * bump_pct
    
    shock_up = SpotShock(name="up", spot_id=spot_id, bump=bump_pct, bump_mode="relative")
    shock_down = SpotShock(name="down", spot_id=spot_id, bump=-bump_pct, bump_mode="relative")
    
    pv_up = pricer.price(portfolio, shock_up.apply(market)).totals.pv
    pv_down = pricer.price(portfolio, shock_down.apply(market)).totals.pv
    pv_base = pricer.price(portfolio, market).totals.pv
    
    gamma = (pv_up - 2*pv_base + pv_down) / (h**2)
    return gamma

def bump_and_reprice_vega(portfolio, market, pricer, vol_id, bump_abs=0.01):
    """Compute vega via central difference."""
    shock_up = VolShock(name="up", vol_id=vol_id, bump=bump_abs, bump_mode="absolute")
    shock_down = VolShock(name="down", vol_id=vol_id, bump=-bump_abs, bump_mode="absolute")
    
    pv_up = pricer.price(portfolio, shock_up.apply(market)).totals.pv
    pv_down = pricer.price(portfolio, shock_down.apply(market)).totals.pv
    
    vega = (pv_up - pv_down) / (2 * bump_abs)
    return vega

# Compute Greeks
fd_delta = bump_and_reprice_delta(portfolio, market, portfolio_pricer, EURUSD_SPOT, 0.01)
fd_gamma = bump_and_reprice_gamma(portfolio, market, portfolio_pricer, EURUSD_SPOT, 0.01)
fd_vega = bump_and_reprice_vega(portfolio, market, portfolio_pricer, EURUSD_VOL, 0.01)

print(f"\nBump-and-Reprice Greeks (1% bump):")
print(f"  Delta: {fd_delta:,.4f}")
print(f"  Gamma: {fd_gamma:,.4f}")
print(f"  Vega:  {fd_vega:,.4f}")

print(f"\nComparison (FD vs Analytical):")
print(f"{'Greek':<10} {'Analytical':<15} {'FD':<15} {'Diff':<15} {'Diff %':<10}")
print("-" * 65)

greeks_compare = [
    ('Delta', analytic_greeks.get('delta', 0), fd_delta),
    ('Gamma', analytic_greeks.get('gamma', 0), fd_gamma),
    ('Vega', analytic_greeks.get('vega', 0), fd_vega),
]

for name, ana, fd in greeks_compare:
    diff = fd - ana
    diff_pct = (diff / ana * 100) if ana != 0 else 0
    print(f"{name:<10} {ana:<15,.4f} {fd:<15,.4f} {diff:<15,.6f} {diff_pct:<10.4f}%")

# =============================================================================
# 2. Bump Size Sensitivity
# =============================================================================

print("\n" + "=" * 70)
print("2. Bump Size Sensitivity Analysis")
print("=" * 70)

bump_sizes = [0.001, 0.005, 0.01, 0.02, 0.05]
delta_by_bump = []

print(f"\nDelta vs Bump Size:")
print(f"{'Bump %':<10} {'FD Delta':<15} {'Analytical':<15} {'Error %':<10}")
print("-" * 50)

analytical_delta = analytic_greeks.get('delta', 0)

for bump in bump_sizes:
    fd_d = bump_and_reprice_delta(portfolio, market, portfolio_pricer, EURUSD_SPOT, bump)
    error = (fd_d - analytical_delta) / analytical_delta * 100 if analytical_delta != 0 else 0
    delta_by_bump.append((bump, fd_d, error))
    print(f"{bump*100:<10.2f} {fd_d:<15,.4f} {analytical_delta:<15,.4f} {error:<10.4f}%")

print("""
Observation:
- Smaller bumps have higher accuracy but more numerical noise
- Larger bumps have more truncation error but less noise
- Sweet spot typically around 0.5-1% for spot bumps
""")

# =============================================================================
# 3. Using the Sensitivities Engine
# =============================================================================

print("\n" + "=" * 70)
print("3. Using the Sensitivities Engine")
print("=" * 70)

# Configure sensitivities
config = SensitivitiesConfig(
    method="analytic",  # Use analytical Greeks where available
    bumps=BumpConfig(
        spot_rel=0.01,   # 1% for spot
        vol_abs=0.01,    # 1 vol point for vol
        rate_abs=0.0001, # 1bp for rates
    ),
    rho_key_by_curve_id={
        USD_CURVE: "rho_domestic",
        EUR_CURVE: "rho_foreign",
    },
)

# Compute sensitivities
report = compute_sensitivities(
    portfolio=portfolio,
    market=market,
    portfolio_pricer=portfolio_pricer,
    config=config,
    requested_greeks=["delta", "gamma", "vega", "rho_domestic", "rho_foreign"],
)

print(f"\nSensitivities Report (Analytical):")
print(f"{'Greek':<15} {'Value':<15} {'Method':<12} {'Market ID':<20}")
print("-" * 62)

for row in report.rows:
    mid_str = row.key.market_id.key() if row.key.market_id else "N/A"
    print(f"{row.key.greek:<15} {row.value:<15,.4f} {row.method:<12} {mid_str:<20}")

# Now with FD method
config_fd = SensitivitiesConfig(
    method="fd_central",
    bumps=BumpConfig(spot_rel=0.01, vol_abs=0.01, rate_abs=0.0001),
    rho_key_by_curve_id={USD_CURVE: "rho_domestic", EUR_CURVE: "rho_foreign"},
)

report_fd = compute_sensitivities(
    portfolio=portfolio,
    market=market,
    portfolio_pricer=portfolio_pricer,
    config=config_fd,
    requested_greeks=["delta", "gamma", "vega", "rho_domestic", "rho_foreign"],
)

print(f"\nSensitivities Report (FD Central):")
print(f"{'Greek':<15} {'Value':<15} {'Bump':<12}")
print("-" * 42)

for row in report_fd.rows:
    bump_str = f"{row.bump:.4f}" if row.bump else "N/A"
    print(f"{row.key.greek:<15} {row.value:<15,.4f} {bump_str:<12}")

# =============================================================================
# 4. Dollar Greeks
# =============================================================================

print("\n" + "=" * 70)
print("4. Dollar Greeks (Scenario P&L)")
print("=" * 70)

delta = analytic_greeks.get('delta', 0)
gamma = analytic_greeks.get('gamma', 0)
vega = analytic_greeks.get('vega', 0)

print(f"""
Dollar Greeks translate sensitivities to P&L for market moves:

  Δ P&L (1% spot move) ≈ Delta × Spot × 0.01 + 0.5 × Gamma × (Spot × 0.01)²

Position:
  Delta = {delta:,.2f}
  Gamma = {gamma:,.2f}
  Vega  = {vega:,.2f}

Scenario P&L Estimates:
""")

scenarios = [
    ("Spot +1%", 0.01, 0),
    ("Spot -1%", -0.01, 0),
    ("Spot +5%", 0.05, 0),
    ("Vol +1pt", 0, 0.01),
    ("Vol -1pt", 0, -0.01),
]

print(f"{'Scenario':<15} {'Delta P&L':<15} {'Gamma P&L':<15} {'Vega P&L':<15} {'Total':<15}")
print("-" * 75)

for name, spot_move, vol_move in scenarios:
    delta_pnl = delta * spot * spot_move if spot_move else 0
    gamma_pnl = 0.5 * gamma * (spot * spot_move)**2 if spot_move else 0
    vega_pnl = vega * vol_move if vol_move else 0
    total = delta_pnl + gamma_pnl + vega_pnl
    
    print(f"{name:<15} {delta_pnl:>+12,.2f}   {gamma_pnl:>+12,.2f}   {vega_pnl:>+12,.2f}   {total:>+12,.2f}")

# =============================================================================
# 5. Visualization
# =============================================================================

print("\n" + "=" * 70)
print("5. Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Bump size convergence
ax = axes[0, 0]
bumps_plot = [b*100 for b, _, _ in delta_by_bump]
errors_plot = [abs(e) for _, _, e in delta_by_bump]

ax.semilogy(bumps_plot, errors_plot, 'o-', color='#2E86AB', linewidth=2, markersize=8)
ax.set_xlabel('Bump Size (%)')
ax.set_ylabel('|Error| (%)')
ax.set_title('FD Delta Error vs Bump Size')
ax.grid(True, alpha=0.3)

# Plot 2: Greeks comparison bar chart
ax = axes[0, 1]
greek_names = ['Delta', 'Gamma\n(×1000)', 'Vega']
ana_values = [analytic_greeks.get('delta', 0), 
              analytic_greeks.get('gamma', 0) * 1000,
              analytic_greeks.get('vega', 0)]
fd_values = [fd_delta, fd_gamma * 1000, fd_vega]

x = np.arange(len(greek_names))
width = 0.35

ax.bar(x - width/2, ana_values, width, label='Analytical', color='#2E86AB')
ax.bar(x + width/2, fd_values, width, label='FD (1%)', color='#E94F37')
ax.set_xticks(x)
ax.set_xticklabels(greek_names)
ax.set_ylabel('Greek Value')
ax.set_title('Analytical vs FD Greeks')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Delta profile
ax = axes[1, 0]
spot_range = np.linspace(spot * 0.85, spot * 1.15, 50)
deltas = []

for s in spot_range:
    shocked_market = Market(
        asof=market.asof,
        quotes={EURUSD_SPOT: Quote(value=s)},
        curves=market.curves,
        vols=market.vols,
    )
    greeks = bsm_pricer.greeks(option, shocked_market)
    deltas.append(greeks.get('delta', 0))

ax.plot(spot_range, deltas, color='#2E86AB', linewidth=2)
ax.axvline(spot, color='gray', linestyle='--', alpha=0.7, label=f'Current: {spot}')
ax.axvline(option.strike, color='red', linestyle=':', alpha=0.7, label=f'Strike: {option.strike}')
ax.set_xlabel('Spot')
ax.set_ylabel('Delta')
ax.set_title('Delta vs Spot (shows Gamma shape)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Vega profile by expiry
ax = axes[1, 1]
expiries = np.linspace(0.1, 2.0, 30)
vegas = []

for exp in expiries:
    temp_option = EuropeanFxVanillaOption(
        ccy_pair="EURUSD", strike=option.strike, expiry_years=exp, is_call=True,
        notional=option.notional, spot_id=EURUSD_SPOT, domestic_curve_id=USD_CURVE,
        foreign_curve_id=EUR_CURVE, vol_id=EURUSD_VOL,
    )
    greeks = bsm_pricer.greeks(temp_option, market)
    vegas.append(greeks.get('vega', 0))

ax.plot(expiries, np.array(vegas)/1000, color='#8B5CF6', linewidth=2)
ax.axvline(option.expiry_years, color='gray', linestyle='--', alpha=0.7, 
           label=f'Current expiry: {option.expiry_years}Y')
ax.set_xlabel('Time to Expiry (years)')
ax.set_ylabel('Vega (thousands)')
ax.set_title('Vega vs Expiry (Vega peaks around ATM, decays to expiry)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sensitivities_computation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to sensitivities_computation.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print("""
Key Takeaways:

1. Greek Computation Methods:
   - Analytical: Exact, fast, but model-specific
   - FD (bump-and-reprice): Universal, but slower

2. Bump Size Trade-off:
   - Small bumps: More accurate, more numerical noise
   - Large bumps: More truncation error, less noise
   - Optimal: ~0.5-1% for spot, ~1pt for vol

3. Sensitivities Engine:
   - compute_sensitivities() handles routing
   - Configurable via SensitivitiesConfig
   - Returns structured SensitivitiesReport

4. Dollar Greeks:
   - Delta P&L ≈ Δ × S × ds
   - Gamma P&L ≈ 0.5 × Γ × (S × ds)²
   - Vega P&L ≈ ν × dσ

5. Greek Profiles:
   - Delta increases towards ITM (S → ∞)
   - Gamma peaks at ATM
   - Vega peaks for longer expiries, ATM

Next: See 03_stress_testing.py for advanced stress scenarios.
""")
