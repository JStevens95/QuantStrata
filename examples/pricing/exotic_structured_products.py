#!/usr/bin/env python3
"""
Exotic Structured Products: Cliquet, Autocallable, Range Accrual

Demonstrates pricing of three path-dependent exotic products using
the library's Monte Carlo pricers and instrument/market data types.

- Cliquet: capped/floored periodic returns (equity)
- Autocallable: early redemption + coupon + put (equity)
- Range Accrual: accrual when rate in range (IR, Hull-White)

Run from project root:
  python examples/pricing/exotic_structured_products.py
"""

from __future__ import annotations

from datetime import date


def run_cliquet_example() -> None:
    """Price an equity cliquet option."""
    from src.instruments.equity.options.cliquet import EquityCliquetOption
    from src.pricers.equity.cliquet_gbm_mc import (
        CliquetMarketData,
        CliquetMcConfig,
        EquityCliquetGbmMcPricer,
    )

    start = date(2025, 1, 1)
    end = date(2026, 1, 1)
    reset_dates = [date(2025, m, 1) for m in range(1, 13)]

    cliquet = EquityCliquetOption(
        underlying_id="SPY",
        notional=1_000_000,
        start_date=start,
        end_date=end,
        reset_dates=reset_dates,
        local_cap=0.03,
        local_floor=-0.01,
        global_cap=0.20,
        global_floor=0.0,
        participation=1.0,
    )

    market = CliquetMarketData(
        spot=100.0,
        volatility=0.20,
        risk_free_rate=0.05,
        dividend_yield=0.02,
        valuation_date=start,
    )

    config = CliquetMcConfig(n_paths=25_000, seed=42, compute_greeks=True)
    pricer = EquityCliquetGbmMcPricer(config=config)
    result = pricer.price(cliquet, market)

    print("  Cliquet (equity, GBM MC)")
    print(f"    Price:  {result.price:,.2f}")
    print(f"    SE:     {result.standard_error:.4f}")
    if result.delta is not None:
        print(f"    Delta:  {result.delta:.4f}")
    if result.vega is not None:
        print(f"    Vega:   {result.vega:,.0f}")
    print(f"    Paths:  {result.n_paths}, time: {result.elapsed_seconds:.2f}s")


def run_autocallable_example() -> None:
    """Price an equity autocallable option."""
    from src.instruments.equity.options.autocallable import EquityAutocallableOption
    from src.pricers.equity.autocallable_gbm_mc import (
        AutocallableMarketData,
        EquityAutocallableGbmMcPricer,
    )

    observation_dates = [
        date(2025, 4, 1),
        date(2025, 7, 1),
        date(2025, 10, 1),
        date(2026, 1, 1),
    ]

    autocall = EquityAutocallableOption(
        underlying_id="SPY",
        notional=1_000_000,
        start_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        observation_dates=observation_dates,
        autocall_barrier=1.0,
        coupon_barrier=0.8,
        put_barrier=0.7,
        coupon_rate=0.10,
    )

    market = AutocallableMarketData(
        spot=100.0,
        volatility=0.22,
        risk_free_rate=0.05,
        dividend_yield=0.02,
        valuation_date=date(2025, 1, 1),
    )

    pricer = EquityAutocallableGbmMcPricer(n_paths=25_000, seed=42)
    result = pricer.price(autocall, market)

    print("  Autocallable (equity, GBM MC)")
    print(f"    Price:  {result.price:,.2f}")
    print(f"    SE:     {result.standard_error:.4f}")
    print(f"    Paths:  {result.n_paths}, time: {result.elapsed_seconds:.2f}s")


def run_range_accrual_example() -> None:
    """Price an IR range accrual note."""
    from src.instruments.ir.options.range_accrual import (
        IrRangeAccrualNote,
        ObservationFrequency,
    )
    from src.pricers.ir.range_accrual_hw_mc import (
        IrRangeAccrualHwMcPricer,
        RangeAccrualMarketData,
    )

    note = IrRangeAccrualNote(
        notional=1_000_000,
        start_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        range_lower=0.03,
        range_upper=0.05,
        accrual_rate=0.06,
        reference_rate_id="USD3M",
        observation_frequency=ObservationFrequency.DAILY,
    )

    market = RangeAccrualMarketData(
        initial_rate=0.04,
        mean_reversion=0.03,
        volatility=0.01,
        long_term_rate=0.04,
        discount_rate=0.05,
        valuation_date=date(2025, 1, 1),
    )

    pricer = IrRangeAccrualHwMcPricer(n_paths=25_000, seed=42)
    result = pricer.price(note, market)

    print("  Range Accrual (IR, Hull-White MC)")
    print(f"    Price:   {result.price:,.2f}")
    print(f"    SE:      {result.standard_error:.4f}")
    print(f"    E[accr]: {result.expected_accrual_fraction:.4f}")
    print(f"    E[coupon]: {result.expected_coupon:,.2f}")
    print(f"    Paths:   {result.n_paths}, time: {result.elapsed_seconds:.2f}s")


def main() -> None:
    print("=" * 60)
    print("Exotic Structured Products Pricing")
    print("=" * 60)

    print("\n1. Cliquet")
    run_cliquet_example()

    print("\n2. Autocallable")
    run_autocallable_example()

    print("\n3. Range Accrual")
    run_range_accrual_example()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
