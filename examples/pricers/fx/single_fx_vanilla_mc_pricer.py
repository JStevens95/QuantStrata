from __future__ import annotations

import matplotlib.pyplot as plt

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import MarketRequest, Universe
from src.marketdata.providers.synthetic.provider import SyntheticProvider

from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer
from src.pricers.fx.european_mc import FxEuropeanVanillaMcPricer, FxMcSimulation

from src.models.numeric.monte_carlo.estimators import mean_stderr, mean_confidence_interval

# plot utilities
from src.core.reporting.plots.pricers.monte_carlo import (
    empirical_log_stats, gbm_terminal_lognormal_params, plot_terminal_spot_distribution,
    McConvergencePoint, plot_mc_convergence_vs_paths, plot_discounted_payoff_distribution, plot_simulated_paths,
    plot_qq_log_terminal_spots, plot_running_pv_estimate, plot_stderr_scaling_vs_paths
)


def _print_simulation_summary(*, sim: FxMcSimulation, pv_bsm: float) -> None:
    """Print compact checkpoints so you can validate the run before trusting plots."""
    pv_mean = float(sim.discounted_payoffs.mean())
    mean, stderr, n = mean_stderr(sim.discounted_payoffs)
    lo, hi = mean_confidence_interval(mean, stderr)

    print("\n================ MC SIMULATION SUMMARY ================")
    print(f"option_type      : {sim.option_type}")
    print(f"S0, K, T         : {sim.spot0:.6f}, {sim.strike:.6f}, {sim.maturity:.6f}")
    print(f"df_domestic      : {sim.df_domestic:.10f}")
    print(f"drift (r_d-r_f)  : {sim.drift:.8f}")
    print(f"sigma            : {sim.sigma:.8f}")
    print(f"notional         : {sim.notional:,.2f}")

    print("\n--- MC controls ---")
    print(f"n_paths req/eff  : {sim.n_paths_requested:,d} / {sim.n_paths_effective:,d}")
    print(f"n_steps, scheme  : {sim.n_steps}, {sim.scheme}")
    print(f"antithetic, seed : {sim.antithetic}, {sim.seed}")

    print("\n--- PV ---")
    print(f"PV_BSM           : {pv_bsm:,.6f}")
    print(f"PV_MC mean       : {pv_mean:,.6f}")
    print(f"abs error        : {abs(pv_mean - float(pv_bsm)):,.6f}")

    print("\n--- MC stats (discounted payoff samples) ---")
    print(f"n               : {n:,d}")
    print(f"mean            : {mean:,.6f}")
    print(f"stderr          : {stderr:,.6f}")
    print(f"CI95            : [{lo:,.6f}, {hi:,.6f}]")
    print("=======================================================")


def main() -> None:

    # =====================================================================================
    # Market setup
    # =====================================================================================
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD.VOL")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([spot_id, vol_id, rd_id, rf_id]),
        )
    )

    # =====================================================================================
    # Trade setup (ATM call for stable demos)
    # =====================================================================================
    spot0 = float(market.quote(spot_id))
    maturity = 1.0
    strike = spot0

    trade = EuropeanFxVanillaOption(
        option_type="call",
        notional=1_000_000.0,
        strike=strike,
        expiry=maturity,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    # =====================================================================================
    # Benchmark (analytic BSM adapter)
    # =====================================================================================
    pv_bsm = float(FxEuropeanVanillaBsmPricer().price(trade, market))

    # =====================================================================================
    # Monte Carlo run ONCE (this is the key change)
    # =====================================================================================
    mc_base = FxEuropeanVanillaMcPricer(
        n_paths=1_000_000,
        seed=7,
        antithetic=True,
        n_steps=128,          # exact GBM terminal distribution is correct with 1 step
        scheme="exact",
    )

    # Store a small subset of paths only if/when we need path plots later.
    # For Plot 1 we do NOT need paths, so keep it lean:
    sim = mc_base.run(trade, market, store_paths=True, paths_keep=500)

    # =====================================================================================
    # Checkpoints (always print)
    # =====================================================================================
    _print_simulation_summary(sim=sim, pv_bsm=pv_bsm)

    # =====================================================================================
    # PLOT 1: Terminal spot distribution + theoretical lognormal overlay
    # =====================================================================================
    # Extra numeric validation in log-space: For exact GBM, ln(S_T) is normal, so these should line up nicely.
    mu_hat, sig_hat = empirical_log_stats(sim.terminal_spots)
    theo = gbm_terminal_lognormal_params(
        spot0=sim.spot0,
        drift=sim.drift,
        vol=sim.sigma,
        maturity=sim.maturity,
    )

    print("\n--- Log-space checks (ln S_T) ---")
    print(f"empirical mean(log S_T) : {mu_hat:.6f}")
    print(f"empirical std (log S_T) : {sig_hat:.6f}")
    print(f"theory     mean(log S_T): {theo.mu_log:.6f}")
    print(f"theory     std (log S_T): {theo.sigma_log:.6f}")

    plot_terminal_spot_distribution(
        terminal_spots=sim.terminal_spots,
        spot0=sim.spot0,
        drift=sim.drift,
        vol=sim.sigma,
        maturity=sim.maturity,
        bins=70,
        title="GBM terminal spot distribution (simulated) + theoretical lognormal pdf",
    )
    # =====================================================================================
    # PLOT 2: MC convergence vs number of paths (mean ± 95% CI) vs BSM benchmark
    # =====================================================================================
    # Keep everything fixed; vary only path count for a clean convergence study.
    path_grid = [
        5_000, 10_000, 15_000, 20_000, 30_000, 40_000, 60_000, 80_000, 120_000, 250_000, 500_000, 1_000_000
    ]

    points: list[McConvergencePoint] = []

    print("\n--- Convergence grid (mean ± 95% CI) ---")
    for n_paths in path_grid:
        # Re-instantiate the pricer so the configuration is explicit per grid point.
        mc = FxEuropeanVanillaMcPricer(
            n_paths=int(n_paths),
            seed=mc_base.seed,
            antithetic=mc_base.antithetic,
            n_steps=mc_base.n_steps,
            scheme=mc_base.scheme,
        )

        sim_n = mc.run(trade, market, store_paths=False)

        mean, stderr, _ = mean_stderr(sim_n.discounted_payoffs)
        lo, hi = mean_confidence_interval(mean, stderr)

        points.append(
            McConvergencePoint(
                n_paths=int(n_paths),
                pv_mean=float(mean),
                pv_ci_lo=float(lo),
                pv_ci_hi=float(hi),
                pv_stderr=float(stderr),
            )
        )

        print(
            f"n_paths={n_paths:>7,d}  mean={mean:>12,.6f}  "
            f"stderr={stderr:>10,.6f}  CI95=[{lo:,.6f}, {hi:,.6f}]"
        )

    ax = plot_mc_convergence_vs_paths(
        points=points,
        pv_benchmark=pv_bsm,
        benchmark_label="BSM PV",
        title="FX Vanilla MC convergence (mean ± 95% CI) vs BSM",
        xlabel="Number of paths",
        ylabel="PV (domestic)",
        use_log_x=True,
    )
    ax.figure.tight_layout()

    # =====================================================================================
    # PLOT 3: Discounted payoff distribution (domestic, notional-scaled) + PV markers
    # =====================================================================================
    plot_discounted_payoff_distribution(
        discounted_payoffs=sim.discounted_payoffs,
        pv_bsm=pv_bsm,
        bins=70,
        title="Discounted payoff distribution (domestic, notional-scaled) with PV markers",
        show_ci_band=True,
        show_percentiles=True,
        log_y=False,  # flip to True if you want to see tail structure more clearly
    )

    # =====================================================================================
    # PLOT 4: Simulated spot paths (subset)
    # =====================================================================================
    if sim.paths is None:
        raise RuntimeError(
            "PLOT_4_PATHS_SUBSET=True but sim.paths is None. "
            "Run mc.run(..., store_paths=True) with a reasonable paths_keep."
        )

    plot_simulated_paths(
        paths=sim.paths,  # type: ignore[arg-type]
        terminal_spots=sim.terminal_spots,
        maturity=sim.maturity,
        spot0=sim.spot0,
        drift=sim.drift,
        vol=sim.sigma,
        strike=sim.strike,
        bins=70,
        title="FX Vanilla MC diagnostics: paths + terminal distribution",
        max_sample_paths=400
    )

    # =====================================================================================
    # PLOT 5: QQ plot of log terminal spots (diagnostic: should be ~straight line for GBM)
    # =====================================================================================
    plot_qq_log_terminal_spots(
        terminal_spots=sim.terminal_spots,
        title="QQ plot: log S(T) vs Normal (GBM exact)",
        max_points=20_000,
    )

    # =====================================================================================
    # PLOT 6: Running PV estimate (single run) + optional band
    # =====================================================================================
    plot_running_pv_estimate(
        discounted_payoffs=sim.discounted_payoffs,
        pv_benchmark=pv_bsm,
        benchmark_label="BSM PV",
        band_sigma=2.0,
        downsample_to=2500,
        title="Running PV estimate (single run) with ±2×stderr band",
    )

    # =====================================================================================
    # PLOT 7: StdErr scaling vs N (log-log) + 1/sqrt(N) reference
    # =====================================================================================
    plot_stderr_scaling_vs_paths(
        points=points,
        show_reference_slope=True,
        title="MC stderr scaling vs number of paths (log-log)",
    )

    # =====================================================================================
    # Show all plots at the end (cleaner than multiple show() calls)
    # =====================================================================================
    plt.show()

if __name__ == "__main__":
    main()