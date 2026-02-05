"""
Scenario PnL Analysis for FX Vanilla Option.

This example demonstrates the complete workflow for computing scenario PnL:

1. Generate correlated historical scenarios for risk factors:
   - FX.SPOT.EURUSD
   - IR.CURVE.USD (parallel shift)
   - IR.CURVE.EUR (parallel shift)
   - FX.VOL.EURUSD (flat vol shift)

2. Build Market snapshots for each scenario

3. Price an FX vanilla option at each scenario

4. Compute PnL distribution and risk metrics

Risk Factors and Shocks
-----------------------
- FX Spot: Simulated using historical returns (filtered block bootstrap)
- Interest Rates: Parallel shifts to zero curves
- Volatility: Multiplicative shocks to flat vol surface

This pattern is typical for:
- VaR computation
- Stress testing
- Scenario analysis
- Risk attribution

Author: QuantStrata Team
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import date, timedelta

# Market data
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote, Curve, VolSurface
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve, ZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface

# Scenario generation
from src.marketdata.scenarios.timeseries import (
    TimeseriesGenerator,
    TimeseriesConfig,
    RiskFactorSpec,
    GBMDynamicsSpec,
    OUDynamicsSpec,
)

# Instrument and pricer
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScenarioConfig:
    """Configuration for scenario generation."""
    
    # Base market values
    spot_eurusd: float = 1.0850
    rate_usd: float = 0.045       # 4.5% USD rate
    rate_eur: float = 0.025       # 2.5% EUR rate
    vol_eurusd: float = 0.08      # 8% implied vol
    
    # Scenario parameters
    n_scenarios: int = 10000
    horizon_days: int = 10        # 10-day VaR horizon
    
    # Risk factor dynamics
    spot_vol: float = 0.08        # FX spot volatility
    rate_vol: float = 0.005       # Rate volatility (50bp annual)
    vol_of_vol: float = 0.20      # Vol-of-vol (20% relative)
    rate_mean_reversion: float = 0.1  # Mean reversion speed for rates
    
    # Correlations
    # [spot, usd_rate, eur_rate, vol]
    correlation: np.ndarray = None
    
    def __post_init__(self):
        if self.correlation is None:
            # Typical FX correlation structure
            self.correlation = np.array([
                [1.00,  0.10, -0.15, -0.30],  # EURUSD spot
                [0.10,  1.00,  0.60,  0.05],  # USD rate
                [-0.15, 0.60,  1.00,  0.05],  # EUR rate
                [-0.30, 0.05,  0.05,  1.00],  # Vol
            ])


@dataclass
class OptionConfig:
    """Configuration for the FX option."""
    
    option_type: str = "call"
    strike: float = 1.10          # Strike price
    expiry_years: float = 0.25    # 3 month expiry
    notional: float = 1_000_000   # 1M EUR notional


# =============================================================================
# Market Snapshot Builder
# =============================================================================

class MarketSnapshotBuilder:
    """
    Builds Market snapshots from scenario values.
    
    Takes scenario values (spot, rate shifts, vol multiplier) and
    constructs full Market objects suitable for pricing.
    """
    
    def __init__(self, base_config: ScenarioConfig):
        self.base = base_config
        
        # Define MarketIds
        self.spot_id = MarketId("FX", "SPOT", "EURUSD")
        self.vol_id = MarketId("FX", "VOL", "EURUSD")
        self.usd_curve_id = MarketId("IR", "CURVE", "USD")
        self.eur_curve_id = MarketId("IR", "CURVE", "EUR")
    
    def build_base_market(self, asof: str = "2024-01-15") -> Market:
        """Build the base (unshocked) market."""
        return self._build_market(
            asof=asof,
            spot=self.base.spot_eurusd,
            rate_usd=self.base.rate_usd,
            rate_eur=self.base.rate_eur,
            vol=self.base.vol_eurusd,
        )
    
    def build_scenario_market(
        self,
        asof: str,
        spot: float,
        rate_usd_shift: float,
        rate_eur_shift: float,
        vol_multiplier: float,
    ) -> Market:
        """
        Build a scenario market with shocked values.
        
        Parameters
        ----------
        asof : str
            As-of date.
        spot : float
            Scenario FX spot.
        rate_usd_shift : float
            Parallel shift to USD rate (absolute, e.g., +0.001 = +10bp).
        rate_eur_shift : float
            Parallel shift to EUR rate.
        vol_multiplier : float
            Multiplicative shock to vol (e.g., 1.1 = +10% vol).
        
        Returns
        -------
        Market
            Scenario market snapshot.
        """
        return self._build_market(
            asof=asof,
            spot=spot,
            rate_usd=self.base.rate_usd + rate_usd_shift,
            rate_eur=self.base.rate_eur + rate_eur_shift,
            vol=self.base.vol_eurusd * vol_multiplier,
        )
    
    def _build_market(
        self,
        asof: str,
        spot: float,
        rate_usd: float,
        rate_eur: float,
        vol: float,
    ) -> Market:
        """Build Market object from raw values."""
        # Quotes
        quotes = {
            self.spot_id: Quote(value=spot),
        }
        
        # Curves (flat zero rate curves for simplicity)
        curves = {
            self.usd_curve_id: FlatZeroRateCurve(continuously_compounded_rate=rate_usd),
            self.eur_curve_id: FlatZeroRateCurve(continuously_compounded_rate=rate_eur),
        }
        
        # Vol surface (flat vol for simplicity)
        vols = {
            self.vol_id: FlatVolSurface(sigma=vol),
        }
        
        return Market(asof=asof, quotes=quotes, curves=curves, vols=vols)


# =============================================================================
# Scenario Generator
# =============================================================================

class FXOptionScenarioGenerator:
    """
    Generates correlated scenarios for FX option risk factors.
    
    Uses Monte Carlo simulation with:
    - GBM for FX spot
    - Ornstein-Uhlenbeck for interest rates (mean-reverting)
    - GBM for implied volatility (geometric for positivity)
    """
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
    
    def generate(self, seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Generate correlated scenario paths.
        
        Returns
        -------
        dict
            Dictionary with arrays for each risk factor:
            - 'spot': FX spot paths, shape (n_time + 1, n_scenarios)
            - 'usd_rate_shift': USD rate shifts
            - 'eur_rate_shift': EUR rate shifts  
            - 'vol_multiplier': Vol multipliers
        """
        cfg = self.config
        dt = 1.0 / 252  # Daily steps
        n_steps = cfg.horizon_days
        
        # Define risk factors
        factors = [
            # FX Spot - GBM
            RiskFactorSpec(
                market_id=MarketId("FX", "SPOT", "EURUSD"),
                initial_value=cfg.spot_eurusd,
                dynamics=GBMDynamicsSpec(
                    drift=cfg.rate_usd - cfg.rate_eur,  # Forward drift
                    vol=cfg.spot_vol,
                ),
                name="EURUSD Spot",
            ),
            # USD Rate Shift - OU (mean reverting to 0)
            RiskFactorSpec(
                market_id=MarketId("IR", "SHIFT", "USD"),
                initial_value=0.0,  # Start at no shift
                dynamics=OUDynamicsSpec(
                    mean=0.0,
                    kappa=cfg.rate_mean_reversion,
                    vol=cfg.rate_vol,
                ),
                name="USD Rate Shift",
            ),
            # EUR Rate Shift - OU
            RiskFactorSpec(
                market_id=MarketId("IR", "SHIFT", "EUR"),
                initial_value=0.0,
                dynamics=OUDynamicsSpec(
                    mean=0.0,
                    kappa=cfg.rate_mean_reversion,
                    vol=cfg.rate_vol,
                ),
                name="EUR Rate Shift",
            ),
            # Vol Multiplier - GBM (geometric for positivity)
            RiskFactorSpec(
                market_id=MarketId("FX", "VOL_MULT", "EURUSD"),
                initial_value=1.0,  # Start at 1x
                dynamics=GBMDynamicsSpec(
                    drift=0.0,
                    vol=cfg.vol_of_vol,
                ),
                name="Vol Multiplier",
            ),
        ]
        
        # Build timeseries config
        start_date = "2024-01-15"
        # Calculate end date
        start = date.fromisoformat(start_date)
        end = start + timedelta(days=n_steps + 5)  # Add buffer for weekends
        
        ts_config = TimeseriesConfig(
            factors=factors,
            correlation=cfg.correlation,
            start_date=start_date,
            end_date=end.isoformat(),
            freq="B",  # Business days
            n_scenarios=cfg.n_scenarios,
        )
        
        # Generate paths
        generator = TimeseriesGenerator(ts_config)
        result = generator.generate_paths(seed=seed)
        
        # Extract paths (take terminal values for simplicity, or all paths)
        spot_id = factors[0].market_id
        usd_shift_id = factors[1].market_id
        eur_shift_id = factors[2].market_id
        vol_mult_id = factors[3].market_id
        
        return {
            'spot': result.paths[spot_id],
            'usd_rate_shift': result.paths[usd_shift_id],
            'eur_rate_shift': result.paths[eur_shift_id],
            'vol_multiplier': result.paths[vol_mult_id],
            'dates': result.dates,
        }


# =============================================================================
# PnL Calculator
# =============================================================================

class ScenarioPnLCalculator:
    """
    Calculates PnL across scenarios.
    
    Computes:
    - Base price (today's value)
    - Scenario prices (value under each scenario)
    - PnL = Scenario price - Base price
    - Risk metrics (VaR, ES, etc.)
    """
    
    def __init__(
        self,
        option: FxVanillaEuropeanOption,
        market_builder: MarketSnapshotBuilder,
    ):
        self.option = option
        self.market_builder = market_builder
        self.pricer = FxVanillaEuropeanOptionBsmPricer()
    
    def compute_base_price(self, asof: str = "2024-01-15") -> float:
        """Compute base (unshocked) price."""
        base_market = self.market_builder.build_base_market(asof)
        return self.pricer.price(self.option, base_market)
    
    def compute_scenario_pnl(
        self,
        scenarios: Dict[str, np.ndarray],
        time_idx: int = -1,  # Use terminal values by default
    ) -> Dict[str, np.ndarray]:
        """
        Compute PnL for all scenarios.
        
        Parameters
        ----------
        scenarios : dict
            Output from FXOptionScenarioGenerator.generate().
        time_idx : int
            Time index to use (-1 for terminal).
        
        Returns
        -------
        dict
            Dictionary with:
            - 'base_price': Base price
            - 'scenario_prices': Array of scenario prices
            - 'pnl': Array of PnL values
            - 'risk_metrics': Dict of risk metrics
        """
        # Compute base price
        base_price = self.compute_base_price()
        
        # Extract scenario values at time_idx
        n_scenarios = scenarios['spot'].shape[1]
        
        spot_scenarios = scenarios['spot'][time_idx, :]
        usd_shifts = scenarios['usd_rate_shift'][time_idx, :]
        eur_shifts = scenarios['eur_rate_shift'][time_idx, :]
        vol_mults = scenarios['vol_multiplier'][time_idx, :]
        
        # Compute scenario prices
        asof = scenarios['dates'][time_idx] if time_idx >= 0 else scenarios['dates'][-1]
        scenario_prices = np.zeros(n_scenarios)
        
        for i in range(n_scenarios):
            scenario_market = self.market_builder.build_scenario_market(
                asof=asof,
                spot=spot_scenarios[i],
                rate_usd_shift=usd_shifts[i],
                rate_eur_shift=eur_shifts[i],
                vol_multiplier=vol_mults[i],
            )
            
            # Adjust option expiry for time elapsed
            # For simplicity, assume we're computing 1-day PnL
            # In production, you'd adjust the expiry
            scenario_prices[i] = self.pricer.price(self.option, scenario_market)
        
        # Compute PnL
        pnl = scenario_prices - base_price
        
        # Compute risk metrics
        risk_metrics = self._compute_risk_metrics(pnl, base_price)
        
        return {
            'base_price': base_price,
            'scenario_prices': scenario_prices,
            'pnl': pnl,
            'risk_metrics': risk_metrics,
            'scenario_spot': spot_scenarios,
            'scenario_vol_mult': vol_mults,
        }
    
    def _compute_risk_metrics(
        self,
        pnl: np.ndarray,
        base_price: float,
    ) -> Dict[str, float]:
        """Compute standard risk metrics from PnL distribution."""
        return {
            'mean_pnl': float(np.mean(pnl)),
            'std_pnl': float(np.std(pnl)),
            'var_95': float(np.percentile(pnl, 5)),  # 95% VaR (5th percentile)
            'var_99': float(np.percentile(pnl, 1)),  # 99% VaR (1st percentile)
            'es_95': float(np.mean(pnl[pnl <= np.percentile(pnl, 5)])),  # Expected shortfall
            'es_99': float(np.mean(pnl[pnl <= np.percentile(pnl, 1)])),
            'max_loss': float(np.min(pnl)),
            'max_gain': float(np.max(pnl)),
            'pnl_skew': float(_skewness(pnl)),
            'pnl_kurtosis': float(_kurtosis(pnl)),
        }


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 4) - 3.0)


# =============================================================================
# Main Example
# =============================================================================

def run_scenario_pnl_analysis(
    scenario_config: ScenarioConfig = None,
    option_config: OptionConfig = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run the complete scenario PnL analysis.
    
    Parameters
    ----------
    scenario_config : ScenarioConfig, optional
        Scenario generation configuration.
    option_config : OptionConfig, optional
        Option configuration.
    seed : int
        Random seed.
    verbose : bool
        Print progress and results.
    
    Returns
    -------
    dict
        Complete results including scenarios, prices, PnL, and metrics.
    """
    # Use defaults if not provided
    scenario_config = scenario_config or ScenarioConfig()
    option_config = option_config or OptionConfig()
    
    if verbose:
        print("=" * 70)
        print("FX Option Scenario PnL Analysis")
        print("=" * 70)
        print(f"\nOption: {option_config.option_type.upper()} EURUSD")
        print(f"  Strike: {option_config.strike}")
        print(f"  Expiry: {option_config.expiry_years * 12:.0f} months")
        print(f"  Notional: {option_config.notional:,.0f} EUR")
        print(f"\nBase Market:")
        print(f"  Spot: {scenario_config.spot_eurusd}")
        print(f"  Vol: {scenario_config.vol_eurusd * 100:.1f}%")
        print(f"  USD Rate: {scenario_config.rate_usd * 100:.2f}%")
        print(f"  EUR Rate: {scenario_config.rate_eur * 100:.2f}%")
        print(f"\nScenarios: {scenario_config.n_scenarios:,}")
        print(f"Horizon: {scenario_config.horizon_days} days")
    
    # Step 1: Build market infrastructure
    market_builder = MarketSnapshotBuilder(scenario_config)
    
    # Step 2: Create option
    option = FxVanillaEuropeanOption(
        option_type=option_config.option_type,
        notional=option_config.notional,
        strike=option_config.strike,
        expiry=option_config.expiry_years,
        spot_id=market_builder.spot_id,
        vol_id=market_builder.vol_id,
        domestic_curve_id=market_builder.usd_curve_id,
        foreign_curve_id=market_builder.eur_curve_id,
    )
    
    # Step 3: Generate scenarios
    if verbose:
        print("\n" + "-" * 70)
        print("Generating scenarios...")
    
    scenario_generator = FXOptionScenarioGenerator(scenario_config)
    scenarios = scenario_generator.generate(seed=seed)
    
    if verbose:
        print(f"  Spot paths shape: {scenarios['spot'].shape}")
        print(f"  Terminal spot range: [{scenarios['spot'][-1].min():.4f}, {scenarios['spot'][-1].max():.4f}]")
    
    # Step 4: Compute PnL
    if verbose:
        print("\n" + "-" * 70)
        print("Computing scenario PnL...")
    
    pnl_calculator = ScenarioPnLCalculator(option, market_builder)
    results = pnl_calculator.compute_scenario_pnl(scenarios)
    
    # Step 5: Display results
    if verbose:
        metrics = results['risk_metrics']
        base_price = results['base_price']
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        print(f"\nBase Price: ${base_price:,.2f}")
        
        print(f"\nPnL Distribution:")
        print(f"  Mean PnL:     ${metrics['mean_pnl']:>12,.2f}")
        print(f"  Std PnL:      ${metrics['std_pnl']:>12,.2f}")
        print(f"  Skewness:     {metrics['pnl_skew']:>12.3f}")
        print(f"  Kurtosis:     {metrics['pnl_kurtosis']:>12.3f}")
        
        print(f"\nRisk Metrics:")
        print(f"  VaR (95%):    ${metrics['var_95']:>12,.2f}")
        print(f"  VaR (99%):    ${metrics['var_99']:>12,.2f}")
        print(f"  ES (95%):     ${metrics['es_95']:>12,.2f}")
        print(f"  ES (99%):     ${metrics['es_99']:>12,.2f}")
        print(f"  Max Loss:     ${metrics['max_loss']:>12,.2f}")
        print(f"  Max Gain:     ${metrics['max_gain']:>12,.2f}")
        
        # PnL percentiles
        print(f"\nPnL Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            pnl_p = np.percentile(results['pnl'], p)
            print(f"  {p:>2}th: ${pnl_p:>12,.2f}")
    
    return {
        'scenarios': scenarios,
        'base_price': results['base_price'],
        'scenario_prices': results['scenario_prices'],
        'pnl': results['pnl'],
        'risk_metrics': results['risk_metrics'],
        'option': option,
        'config': {
            'scenario': scenario_config,
            'option': option_config,
        },
    }


# =============================================================================
# Plotting (optional)
# =============================================================================

def plot_results(results: Dict, save_path: str = None):
    """
    Plot scenario PnL results.
    
    Creates:
    - PnL histogram with VaR lines
    - PnL vs Spot scatter
    - PnL vs Vol scatter
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    pnl = results['pnl']
    metrics = results['risk_metrics']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. PnL Histogram
    ax = axes[0]
    ax.hist(pnl, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(metrics['var_95'], color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: ${metrics["var_95"]:,.0f}')
    ax.axvline(metrics['var_99'], color='red', linestyle='--', linewidth=2, label=f'VaR 99%: ${metrics["var_99"]:,.0f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('PnL ($)')
    ax.set_ylabel('Density')
    ax.set_title('PnL Distribution')
    ax.legend(fontsize=8)
    
    # 2. PnL vs Spot (if available in results)
    ax = axes[1]
    if 'scenario_spot' in results:
        spot = results['scenario_spot']
        # Sample for plotting (avoid overplotting)
        n_plot = min(2000, len(pnl))
        idx = np.random.choice(len(pnl), n_plot, replace=False)
        ax.scatter(spot[idx], pnl[idx], alpha=0.3, s=5, c='steelblue')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('EURUSD Spot')
        ax.set_ylabel('PnL ($)')
        ax.set_title('PnL vs Spot')
    
    # 3. PnL vs Vol
    ax = axes[2]
    if 'scenario_vol_mult' in results:
        vol_mult = results['scenario_vol_mult']
        n_plot = min(2000, len(pnl))
        idx = np.random.choice(len(pnl), n_plot, replace=False)
        ax.scatter(vol_mult[idx], pnl[idx], alpha=0.3, s=5, c='steelblue')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Vol Multiplier')
        ax.set_ylabel('PnL ($)')
        ax.set_title('PnL vs Vol')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Run the analysis
    results = run_scenario_pnl_analysis(
        scenario_config=ScenarioConfig(n_scenarios=10000, horizon_days=10),
        option_config=OptionConfig(
            option_type="call",
            strike=1.10,
            expiry_years=0.25,
            notional=1_000_000,
        ),
        seed=42,
        verbose=True,
    )
    
    # Optionally plot
    if "--plot" in sys.argv:
        plot_results(results, save_path="scenario_pnl_fx_option.png")
