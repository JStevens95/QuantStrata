# src/core/reporting/plots/monte_carlo.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from src.models.common.normal import std_normal_ppf
from src.models.numeric.monte_carlo.estimators import mean_stderr, mean_confidence_interval


# --------------------------------------------------------------------------------------
# Param objects
# --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LogNormalParams:
    """
    Parameters for a lognormal distribution in the form:

        ln(S_T) ~ Normal(mu_log, sigma_log^2)

    This is the natural parameterization for GBM terminal distributions.
    """

    mu_log: float
    sigma_log: float


@dataclass(frozen=True, slots=True)
class McConvergencePoint:
    """
    One Monte Carlo convergence datapoint for a particular path count.

    Notes
    -----
    - pv_mean / pv_ci_* should already be in *domestic PV units* (i.e. what you report to users).
    - pv_ci_lo/hi typically come from mean +/- z * stderr (normal approx).
    """
    n_paths: int
    pv_mean: float
    pv_ci_lo: float
    pv_ci_hi: float
    pv_stderr: float


@dataclass(frozen=True, slots=True)
class PayoffStats:
    """Lightweight stats summary for discounted payoff samples."""
    n: int
    mean: float
    stderr: float
    ci95_lo: float
    ci95_hi: float
    p50: float
    p95: float
    p99: float


# --------------------------------------------------------------------------------------
# Core math helpers
# --------------------------------------------------------------------------------------
def _as_sorted_points(points: Iterable[McConvergencePoint]) -> list[McConvergencePoint]:
    """Return points sorted by n_paths, validating basic invariants."""
    pts = list(points)
    if not pts:
        raise ValueError("points must be non-empty.")

    for p in pts:
        if p.n_paths <= 0:
            raise ValueError(f"n_paths must be positive; got {p.n_paths}.")
        if p.pv_ci_hi < p.pv_ci_lo:
            raise ValueError(f"CI invalid for n_paths={p.n_paths}: hi < lo.")
    pts.sort(key=lambda p: p.n_paths)
    return pts


def _extract_xy_ci(points: Sequence[McConvergencePoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert points -> arrays for plotting."""
    x = np.asarray([p.n_paths for p in points], dtype=np.float64)
    y = np.asarray([p.pv_mean for p in points], dtype=np.float64)
    lo = np.asarray([p.pv_ci_lo for p in points], dtype=np.float64)
    hi = np.asarray([p.pv_ci_hi for p in points], dtype=np.float64)
    return x, y, lo, hi

def empirical_log_stats(terminal_spots: np.ndarray) -> Tuple[float, float]:
    """
    Compute empirical (mean, std) of ln(S_T) from simulated terminal spots.

    Parameters
    ----------
    terminal_spots:
        Simulated terminal spot values S_T, shape (n,).

    Returns
    -------
    (mu_hat, sigma_hat):
        mu_hat: mean of ln(S_T)
        sigma_hat: std of ln(S_T) using ddof=1 when n>1 (unbiased sample std)

    Notes
    -----
    Under exact GBM simulation, ln(S_T) should be approximately normal, so these
    should match theory closely for large n.
    """
    s = np.asarray(terminal_spots, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError("terminal_spots must be a 1D array.")
    if s.size == 0:
        raise ValueError("terminal_spots must be non-empty.")
    if np.any(s <= 0.0):
        raise ValueError("terminal_spots must be strictly positive to take logs.")

    log_s = np.log(s)

    mu_hat = float(log_s.mean())

    # Unbiased sample std for stability in small samples.
    if log_s.size <= 1:
        sigma_hat = 0.0
    else:
        sigma_hat = float(log_s.std(ddof=1))

    return mu_hat, sigma_hat


def gbm_terminal_lognormal_params(
    *,
    spot0: float,
    drift: float,
    vol: float,
    maturity: float,
) -> LogNormalParams:
    """
    Return the theoretical lognormal parameters for GBM terminal spot:

        dS/S = drift dt + vol dW

    Then:
        ln(S_T) ~ Normal(
            ln(S0) + (drift - 0.5*vol^2)*T,
            (vol*sqrt(T))^2
        )

    Parameters
    ----------
    spot0:
        Initial spot S0 (>0).
    drift:
        Drift parameter (mu).
    vol:
        Volatility sigma (>=0).
    maturity:
        Time horizon T (>=0).

    Returns
    -------
    LogNormalParams(mu_log, sigma_log)
    """
    if spot0 <= 0.0:
        raise ValueError("spot0 must be > 0.")
    if maturity < 0.0:
        raise ValueError("maturity must be >= 0.")
    if vol < 0.0:
        raise ValueError("vol must be >= 0.")

    t = float(maturity)
    sigma = float(vol)

    mu_log = float(math.log(float(spot0)) + (float(drift) - 0.5 * sigma * sigma) * t)
    sigma_log = float(sigma * math.sqrt(t)) if t > 0.0 else 0.0

    return LogNormalParams(mu_log=mu_log, sigma_log=sigma_log)


def _lognormal_pdf(x: np.ndarray, *, mu_log: float, sigma_log: float) -> np.ndarray:
    """
    Vectorized lognormal PDF for x > 0 given ln(x) ~ N(mu_log, sigma_log^2).

    Returns
    -------
    pdf values with shape matching x.
    """
    x = np.asarray(x, dtype=np.float64)

    # Guard for degenerate sigma (T=0 or vol=0). In that case, distribution collapses
    # at exp(mu_log); a PDF isn’t meaningful. We return zeros for overlay stability.
    if sigma_log <= 0.0:
        return np.zeros_like(x, dtype=np.float64)

    # PDF = 1/(x*sigma*sqrt(2pi)) * exp(-(ln x - mu)^2/(2 sigma^2))
    safe_x = np.maximum(x, np.finfo(np.float64).tiny)  # avoid division by zero
    z = (np.log(safe_x) - float(mu_log)) / float(sigma_log)
    norm_const = 1.0 / (safe_x * float(sigma_log) * math.sqrt(2.0 * math.pi))
    return norm_const * np.exp(-0.5 * z * z)


def summarize_discounted_payoffs(discounted_payoffs: np.ndarray) -> PayoffStats:
    """
    Compute a compact set of summary statistics for discounted payoff samples.

    Parameters
    ----------
    discounted_payoffs:
        1D array of discounted payoff samples in *domestic currency*,
        already scaled by notional.

    Returns
    -------
    PayoffStats
        Mean + standard error + 95% CI + a few percentiles.
    """
    samples = np.asarray(discounted_payoffs, dtype=np.float64).reshape(-1)
    mean, stderr, n = mean_stderr(samples)
    lo, hi = mean_confidence_interval(mean, stderr)

    p50 = float(np.quantile(samples, 0.50))
    p95 = float(np.quantile(samples, 0.95))
    p99 = float(np.quantile(samples, 0.99))

    return PayoffStats(
        n=int(n),
        mean=float(mean),
        stderr=float(stderr),
        ci95_lo=float(lo),
        ci95_hi=float(hi),
        p50=p50,
        p95=p95,
        p99=p99,
    )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def save_or_show_plot(
    *,
    fig: plt.Figure,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 180,
) -> None:
    """
    Small helper so examples can consistently save and/or display plots.

    Parameters
    ----------
    fig:
        Matplotlib figure to save/show.
    save_path:
        If provided, saves the figure to this path.
    show:
        If True, displays the plot window (plt.show()).
    dpi:
        Save DPI if saving.
    """
    if save_path:
        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _running_mean_and_stderr(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute running mean and running standard error for i=1..n.

    Parameters
    ----------
    samples:
        1D array of samples (e.g., discounted payoff PV samples).

    Returns
    -------
    (running_mean, running_stderr):
        running_mean[i-1]  = mean(samples[:i])
        running_stderr[i-1]= std(samples[:i], ddof=1)/sqrt(i)   for i>=2
                            0 for i==1
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("samples must be non-empty.")
    if np.any(~np.isfinite(x)):
        raise ValueError("samples contains non-finite values.")

    n = x.size
    idx = np.arange(1, n + 1, dtype=np.float64)

    csum = np.cumsum(x, dtype=np.float64)
    csum2 = np.cumsum(x * x, dtype=np.float64)

    running_mean = csum / idx

    # Sample variance with ddof=1:
    # var = (sum(x^2) - n*mean^2) / (n-1)
    var_num = csum2 - idx * (running_mean * running_mean)
    running_var = np.zeros(n, dtype=np.float64)
    valid = idx >= 2.0
    running_var[valid] = np.maximum(var_num[valid] / (idx[valid] - 1.0), 0.0)

    running_std = np.sqrt(running_var)
    running_stderr = np.zeros(n, dtype=np.float64)
    running_stderr[valid] = running_std[valid] / np.sqrt(idx[valid])

    return running_mean, running_stderr


def _validate_paths_matrix(paths: np.ndarray) -> np.ndarray:
    """
    Validate and normalize a simulated paths matrix.

    Parameters
    ----------
    paths:
        Array of simulated paths, expected shape (n_paths, n_times).
        - n_times should be >= 2 (includes t=0 column and at least one step)

    Returns
    -------
    np.ndarray
        A float64 array of shape (n_paths, n_times).

    Raises
    ------
    ValueError
        If paths is not a 2D array or contains non-finite values.
    """
    x = np.asarray(paths, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"paths must be 2D (n_paths, n_times); got ndim={x.ndim}.")
    if x.shape[0] <= 0 or x.shape[1] <= 1:
        raise ValueError(f"paths must have shape (n_paths>0, n_times>=2); got {x.shape}.")
    if np.any(~np.isfinite(x)):
        raise ValueError("paths contains non-finite values.")
    return x


def _build_time_grid(*, maturity: float, n_times: int) -> np.ndarray:
    """
    Build an equally spaced time grid matching the stored paths.

    Parameters
    ----------
    maturity:
        Final time horizon T.
    n_times:
        Number of time points stored in paths (columns). This equals n_steps + 1.

    Returns
    -------
    np.ndarray
        Time grid of shape (n_times,) from 0 to maturity.
    """
    if maturity < 0.0:
        raise ValueError("maturity must be >= 0.")
    if n_times <= 1:
        raise ValueError("n_times must be >= 2.")
    return np.linspace(0.0, float(maturity), int(n_times), dtype=np.float64)


# --------------------------------------------------------------------------------------
# Plot: Terminal spot distribution with theoretical overlay
# --------------------------------------------------------------------------------------

def plot_terminal_spot_distribution(
    *,
    terminal_spots: np.ndarray,
    spot0: float,
    drift: float,
    vol: float,
    maturity: float,
    bins: int = 70,
    title: str = "Terminal spot distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the simulated terminal spot distribution (histogram) and overlay the
    theoretical GBM lognormal PDF.

    Parameters
    ----------
    terminal_spots:
        Simulated S(T) values, shape (n,).
    spot0, drift, vol, maturity:
        Parameters used for the theoretical GBM lognormal overlay.
    bins:
        Histogram bin count.
    title:
        Plot title.
    ax:
        Optional matplotlib Axes. If None, a new figure+axes is created.

    Returns
    -------
    matplotlib Axes
        The axes containing the plot.

    Notes
    -----
    - We plot the histogram as a density so it is directly comparable to the PDF.
    - The overlay uses a robust x-range based on percentiles to avoid extreme tails
      dominating the axis limits.
    """
    s_t = np.asarray(terminal_spots, dtype=np.float64)
    if s_t.ndim != 1:
        raise ValueError("terminal_spots must be a 1D array.")
    if s_t.size == 0:
        raise ValueError("terminal_spots must be non-empty.")
    if np.any(s_t <= 0.0):
        raise ValueError("terminal_spots must be strictly positive.")

    # Create axes if not provided.
    if ax is None:
        _, ax = plt.subplots()

    # Robust plotting range (avoid a single extreme tail point ruining the view).
    x_lo = float(np.percentile(s_t, 0.5))
    x_hi = float(np.percentile(s_t, 99.5))
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_lo = float(s_t.min())
        x_hi = float(s_t.max())

    # Histogram (density=True) so we can overlay a PDF.
    ax.hist(
        s_t,
        bins=int(bins),
        density=True,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.5,
        label="Simulated S(T) density",
    )

    # Theoretical lognormal overlay.
    theo = gbm_terminal_lognormal_params(spot0=spot0, drift=drift, vol=vol, maturity=maturity)
    x_grid = np.linspace(max(1e-12, x_lo), x_hi, 600)
    pdf = _lognormal_pdf(x_grid, mu_log=theo.mu_log, sigma_log=theo.sigma_log)

    ax.plot(
        x_grid,
        pdf,
        linewidth=2.0,
        label="Theoretical lognormal PDF (GBM)",
    )

    # Helpful reference lines.
    ax.axvline(float(spot0), linewidth=1.5, linestyle="--", label="S0")
    ax.axvline(float(s_t.mean()), linewidth=1.5, linestyle=":", label="Mean(S(T))")

    # Labels and styling.
    ax.set_title(str(title))
    ax.set_xlabel("Terminal spot S(T)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    return ax


def plot_mc_convergence_vs_paths(
    *,
    points: Iterable[McConvergencePoint],
    pv_benchmark: Optional[float] = None,
    benchmark_label: str = "BSM PV",
    title: str = "MC convergence (mean ± 95% CI) vs number of paths",
    xlabel: str = "Number of paths",
    ylabel: str = "PV (domestic)",
    use_log_x: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot Monte Carlo convergence: PV mean with CI band vs path count.

    Parameters
    ----------
    points:
        Convergence points for different `n_paths`.
    pv_benchmark:
        Optional benchmark PV (e.g. BSM analytic) drawn as a horizontal line.
    benchmark_label:
        Legend label for the benchmark line.
    title, xlabel, ylabel:
        Plot labelling.
    use_log_x:
        If True, sets x-axis to log scale (recommended).
    ax:
        Optional matplotlib Axes to plot into. If None, a new figure+axes is created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    - For a fair convergence study, keep seed / scheme / antithetic / n_steps fixed,
      and vary only `n_paths`.
    """
    pts = _as_sorted_points(points)
    x, y, lo, hi = _extract_xy_ci(pts)

    if ax is None:
        _, ax = plt.subplots()

    # Mean line + markers
    ax.plot(x, y, marker="o", linewidth=1.5, label="MC mean")

    # CI band
    ax.fill_between(x, lo, hi, alpha=0.25, label="95% CI")

    # Optional benchmark line
    if pv_benchmark is not None:
        ax.axhline(float(pv_benchmark), linewidth=1.5, linestyle="--", label=str(benchmark_label))

    # Axis formatting
    if use_log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Grid + legend
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.legend()

    return ax


def plot_discounted_payoff_distribution(
    *,
    discounted_payoffs: np.ndarray,
    pv_bsm: Optional[float] = None,
    bins: int = 70,
    title: str = "Discounted payoff distribution (domestic)",
    show_ci_band: bool = True,
    show_percentiles: bool = True,
    log_y: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a histogram of discounted payoff samples (domestic), with PV markers.

    What this validates
    -------------------
    - You are plotting the correct random variable for pricing:
        PV = mean(discounted_payoffs)
    - Shape intuition (mass at 0, heavy right tail for calls, etc.)
    - MC vs BSM benchmark alignment.

    Parameters
    ----------
    discounted_payoffs:
        1D array of discounted payoff samples in domestic currency,
        already scaled by notional.
    pv_bsm:
        Optional analytic PV benchmark to overlay as a vertical line.
    bins:
        Histogram bin count.
    title:
        Plot title.
    show_ci_band:
        If True, shade the 95% CI band around the MC mean.
    show_percentiles:
        If True, add vertical markers for p50/p95/p99.
    log_y:
        If True, plot histogram counts on a log scale (useful for heavy tails).

    Returns
    -------
    (fig, ax)
        Matplotlib Figure and Axes for further customization or saving.
    """
    samples = np.asarray(discounted_payoffs, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        raise ValueError("discounted_payoffs must be non-empty.")
    if np.any(~np.isfinite(samples)):
        raise ValueError("discounted_payoffs contains non-finite values.")

    stats = summarize_discounted_payoffs(samples)

    # -----------------------------
    # Figure / axes
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    # Histogram: density=False because “count” is intuitive for MC diagnostics.
    ax.hist(samples, bins=int(bins), alpha=0.85)

    # Main PV markers
    ax.axvline(stats.mean, linewidth=2.0, linestyle="-", label=f"MC mean PV = {stats.mean:,.2f}")
    if pv_bsm is not None:
        ax.axvline(float(pv_bsm), linewidth=2.0, linestyle="--", label=f"BSM PV = {float(pv_bsm):,.2f}")

    # Confidence interval band for the mean (not the distribution)
    if show_ci_band and stats.n > 1:
        ax.axvspan(stats.ci95_lo, stats.ci95_hi, alpha=0.18, label="95% CI (mean)")

    # Distribution percentiles (useful for tails)
    if show_percentiles:
        ax.axvline(stats.p50, linewidth=1.5, linestyle=":", label=f"P50 = {stats.p50:,.2f}")
        ax.axvline(stats.p95, linewidth=1.5, linestyle=":", label=f"P95 = {stats.p95:,.2f}")
        ax.axvline(stats.p99, linewidth=1.5, linestyle=":", label=f"P99 = {stats.p99:,.2f}")

    # Cosmetics
    ax.set_title(title)
    ax.set_xlabel("Discounted payoff sample (domestic PV, notional-scaled)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    if log_y:
        ax.set_yscale("log")

    # Summary text box (kept compact and interview-friendly)
    summary_lines = [
        f"n = {stats.n:,d}",
        f"mean = {stats.mean:,.2f}",
        f"stderr = {stats.stderr:,.2f}",
        f"CI95 = [{stats.ci95_lo:,.2f}, {stats.ci95_hi:,.2f}]",
    ]
    ax.text(
        0.99,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig, ax


def plot_paths_subset(
    *,
    paths: np.ndarray,
    maturity: float,
    spot0: Optional[float] = None,
    strike: Optional[float] = None,
    max_paths: int = 300,
    title: str = "Simulated spot paths (subset)",
    alpha: float = 0.35,
    linewidth: float = 1.0,
    show_median: bool = True,
    show_mean: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a subset of simulated spot paths over time.

    This is a diagnostics / reporting plot:
    - Validates the simulator is stepping correctly across time
    - Gives intuition about volatility scale and path dispersion
    - Useful for interviews and reports (especially when combined with terminal distribution)

    Parameters
    ----------
    paths:
        Simulated paths array of shape (n_paths, n_steps+1).
        Typically this is `sim.paths` from FxMcSimulation when `store_paths=True`.
    maturity:
        Option maturity T used to label the x-axis time grid.
    spot0:
        Optional S0 to draw as a reference horizontal line (if provided).
    strike:
        Optional strike K to draw as a reference horizontal line (if provided).
    max_paths:
        Maximum number of paths to plot (for readability/performance).
    title:
        Plot title.
    alpha:
        Line alpha for individual paths.
    linewidth:
        Line width for individual paths.
    show_median:
        If True, overlay the median path across the plotted subset.
    show_mean:
        If True, overlay the mean path across the plotted subset.
    ax:
        Optional matplotlib Axes. If None, a new figure+axes is created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    - This function *does not* re-simulate anything; it only visualizes stored paths.
    - For large MC runs, store only a subset (e.g. paths_keep=200..2000) and pass that here.
    """
    x = _validate_paths_matrix(paths)

    # Decide how many paths to plot.
    n_paths_total = int(x.shape[0])
    n_paths_plot = min(max(1, int(max_paths)), n_paths_total)
    x_plot = x[:n_paths_plot, :]

    # Time grid aligned with columns.
    t_grid = _build_time_grid(maturity=float(maturity), n_times=int(x_plot.shape[1]))

    # Create axes if not provided.
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.0))

    # Plot individual paths.
    for i in range(n_paths_plot):
        ax.plot(t_grid, x_plot[i, :], alpha=float(alpha), linewidth=float(linewidth))

    # Overlay summary path(s) (computed across plotted subset).
    if show_median:
        median_path = np.median(x_plot, axis=0)
        ax.plot(t_grid, median_path, linewidth=2.5, linestyle="-", label="Median path")

    if show_mean:
        mean_path = np.mean(x_plot, axis=0)
        ax.plot(t_grid, mean_path, linewidth=2.5, linestyle="--", label="Mean path")

    # Reference lines.
    if spot0 is not None:
        ax.axhline(float(spot0), linewidth=1.5, linestyle="--", label="S0")
    if strike is not None:
        ax.axhline(float(strike), linewidth=1.5, linestyle=":", label="Strike K")

    # Cosmetics.
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Spot S(t)")
    ax.grid(True, alpha=0.25)

    # Only show legend if we actually added labelled artists.
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")

    return ax


def plot_simulated_paths(
    *,
    paths: np.ndarray,
    terminal_spots: np.ndarray,
    maturity: float,
    spot0: float,
    drift: float,
    vol: float,
    strike: Optional[float] = None,
    bins: int = 70,
    fan_levels: Tuple[float, float, float, float] = (0.05, 0.25, 0.75, 0.95),
    show_sample_paths: bool = True,
    max_sample_paths: int = 200,
    title: str = "Simulated paths with terminal distribution",
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Single “main” paths chart with a rotated (horizontal) terminal distribution
    attached on the right-hand side.

    Layout
    ------
    - Main axes (left): paths + percentile fan over time.
    - Side axes (right): histogram of terminal S(T), rotated 90° (orientation='horizontal'),
      sharing the y-axis with the main plot so the distribution aligns to spot levels.

    Notes
    -----
    This is often preferable for reports because:
    - You keep one coherent time-series plot
    - You still see the terminal cross-section distribution in the same visual frame
    """

    # -----------------------------
    # Validate inputs
    # -----------------------------
    p = np.asarray(paths, dtype=np.float64)
    s_t = np.asarray(terminal_spots, dtype=np.float64).reshape(-1)

    if p.ndim != 2 or p.shape[1] < 3:
        raise ValueError("paths must be 2D with at least 3 time points (n_steps >= 2).")
    if s_t.size == 0:
        raise ValueError("terminal_spots must be non-empty.")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(s_t)):
        raise ValueError("paths/terminal_spots contain non-finite values.")
    if maturity <= 0.0:
        raise ValueError("maturity must be > 0.")

    # -----------------------------
    # Create main axes
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_title(str(title))
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Spot")
    ax.grid(True, linestyle=":", alpha=0.7)

    # Time grid
    n_paths, n_cols = int(p.shape[0]), int(p.shape[1])
    t_grid = np.linspace(0.0, float(maturity), n_cols)

    # Sample paths in background
    if show_sample_paths and n_paths > 0:
        k = min(int(max_sample_paths), n_paths)
        ax.plot(t_grid, p[:k, :].T, linewidth=0.9, alpha=0.5)

    # Percentile fan
    q_lo_outer, q_lo_inner, q_hi_inner, q_hi_outer = fan_levels
    lo_outer = np.quantile(p, q_lo_outer, axis=0)
    lo_inner = np.quantile(p, q_lo_inner, axis=0)
    hi_inner = np.quantile(p, q_hi_inner, axis=0)
    hi_outer = np.quantile(p, q_hi_outer, axis=0)

    ax.fill_between(t_grid, lo_outer, hi_outer, alpha=0.18, label=f"P{int(q_lo_outer*100)}–P{int(q_hi_outer*100)}")
    ax.fill_between(t_grid, lo_inner, hi_inner, alpha=0.28, label=f"P{int(q_lo_inner*100)}–P{int(q_hi_inner*100)}")

    # Median line
    med = np.quantile(p, 0.5, axis=0)
    ax.plot(t_grid, med, linewidth=2.2, label="Median")

    # Reference levels
    ax.axhline(float(spot0), linewidth=1.6, linestyle="--", label="S0")
    if strike is not None:
        ax.axhline(float(strike), linewidth=1.6, linestyle=":", label="K")

    # -----------------------------
    # Append the side distribution axis (shares y)
    # -----------------------------
    # Import locally to keep dependencies minimal.
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    ax_side = divider.append_axes("right", size="28%", pad=0.12, sharey=ax)

    # Histogram rotated 90°: density along x, spot along y
    ax_side.hist(
        s_t,
        bins=int(bins),
        density=True,
        orientation="horizontal",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
        label="S(T) density",
    )

    # Optional theoretical/fitted overlays (also rotated)
    # We plot density(x) against spot(y): x = pdf(y)
    # Only meaningful if terminal spots are > 0
    if np.all(s_t > 0.0):
        y_lo = float(np.percentile(s_t, 0.5))
        y_hi = float(np.percentile(s_t, 99.5))
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
            y_lo, y_hi = float(s_t.min()), float(s_t.max())

        y_grid = np.linspace(max(1e-12, y_lo), y_hi, 600)

        # Empirical fit
        mu_hat, sig_hat = empirical_log_stats(s_t)
        pdf_fit = _lognormal_pdf(y_grid, mu_log=mu_hat, sigma_log=sig_hat)
        ax_side.plot(pdf_fit, y_grid, linewidth=2.0, label="Fitted lognormal")

        # Theoretical GBM
        theo = gbm_terminal_lognormal_params(spot0=spot0, drift=drift, vol=vol, maturity=maturity)
        pdf_theo = _lognormal_pdf(y_grid, mu_log=theo.mu_log, sigma_log=theo.sigma_log)
        ax_side.plot(pdf_theo, y_grid, linewidth=2.0, linestyle="--", label="Theoretical GBM")

    # Style side axis
    ax_side.set_xlabel("Density")
    ax_side.grid(True, linestyle=":", alpha=0.5)

    # Hide duplicate y tick labels on the right (cleaner)
    plt.setp(ax_side.get_yticklabels(), visible=False)

    # Legends: keep main legend on left; side legend on right (optional)
    ax.legend(loc="upper left", frameon=True)
    ax_side.legend(loc="upper right", frameon=True)

    y_min = float(np.nanmin(p))
    y_max = float(np.nanmax(p))

    # Include reference lines so they don't get clipped
    y_min = min(y_min, float(spot0))
    y_max = max(y_max, float(spot0))
    if strike is not None:
        y_min = min(y_min, float(strike))
        y_max = max(y_max, float(strike))

    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05 * max(abs(y_max), 1.0)
    y_lo = y_min - pad
    y_hi = y_max + pad

    ax.set_ylim(y_lo, y_hi)
    ax_side.set_ylim(y_lo, y_hi)  # explicit even though sharey

    # Lock y autoscaling so nothing later overrides it
    ax.set_autoscale_on(False)
    ax_side.set_autoscale_on(False)

    fig.tight_layout()
    return fig, (ax, ax_side)


def plot_qq_log_terminal_spots(
    *,
    terminal_spots: np.ndarray,
    title: str = "QQ plot: log S(T) vs Normal",
    max_points: int = 20_000,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    QQ plot for ln(S_T) against a fitted Normal distribution.

    Interpretation
    --------------
    - If ln(S_T) is Normal, points fall ~ on a straight line.
    - For exact GBM (scheme='exact'), this is expected (up to sampling noise).
    - Curvature or heavy-tail deviations indicate non-normality (e.g., wrong dynamics,
      discretization issues, or model misspecification).

    Parameters
    ----------
    terminal_spots:
        Simulated terminal spots S_T, shape (n,). Must be > 0.
    title:
        Plot title.
    max_points:
        Subsample cap for speed/clarity when n is very large.
    ax:
        Optional Axes.

    Returns
    -------
    Axes
        Axes containing the QQ plot.
    """
    s_t = np.asarray(terminal_spots, dtype=np.float64).reshape(-1)
    if s_t.size == 0:
        raise ValueError("terminal_spots must be non-empty.")
    if np.any(s_t <= 0.0):
        raise ValueError("terminal_spots must be strictly positive (for log).")

    # Subsample for speed (deterministic: take evenly spaced indices)
    n = int(s_t.size)
    if n > int(max_points):
        idx = np.linspace(0, n - 1, int(max_points)).astype(int)
        s_t = s_t[idx]

    log_s = np.log(s_t)
    log_s.sort()

    # Fit Normal parameters to empirical log data
    mu_hat = float(log_s.mean())
    sig_hat = float(log_s.std(ddof=1)) if log_s.size > 1 else 0.0

    # Theoretical normal quantiles at plotting positions
    m = int(log_s.size)
    p = (np.arange(1, m + 1, dtype=np.float64) - 0.5) / float(m)
    z = std_normal_ppf(p)
    theo = mu_hat + sig_hat * z  # theoretical quantiles for fitted Normal

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6.5))

    ax.scatter(theo, log_s, s=10, alpha=0.65, label="Empirical quantiles")
    # 45-degree reference line in fitted-theory space
    lo = float(min(theo.min(), log_s.min()))
    hi = float(max(theo.max(), log_s.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2.0, label="y = x reference")

    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles (Normal fit to log S(T))")
    ax.set_ylabel("Empirical quantiles (log S(T))")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best")

    # Small annotation with fitted params
    ax.text(
        0.02,
        0.98,
        f"fit: mu={mu_hat:.6f}\nfit: sigma={sig_hat:.6f}\npoints={m:,d}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    return ax


def plot_running_pv_estimate(
    *,
    discounted_payoffs: np.ndarray,
    pv_benchmark: Optional[float] = None,
    benchmark_label: str = "BSM PV",
    title: str = "Running PV estimate vs number of paths",
    xlabel: str = "Number of paths (running)",
    ylabel: str = "PV (domestic)",
    band_sigma: float = 2.0,
    downsample_to: int = 3000,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the running Monte Carlo PV estimate as you add more paths.

    Why this is useful
    ------------------
    - Shows *stability* of the estimate over the course of a single simulation.
    - Very intuitive sanity check for interviews and reporting.

    Parameters
    ----------
    discounted_payoffs:
        1D array of discounted payoff samples in domestic currency (notional-scaled).
    pv_benchmark:
        Optional benchmark PV (e.g., analytic BSM) shown as a horizontal line.
    band_sigma:
        If >0, shows +/- band_sigma * running_stderr as a confidence-style band.
        band_sigma=2.0 is a common visual.
    downsample_to:
        For very large n, plotting every point is slow. We downsample to at most
        this many points (keeping early and late behavior visible).
    ax:
        Optional axis to draw on.

    Returns
    -------
    Axes
    """
    x = np.asarray(discounted_payoffs, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("discounted_payoffs must be non-empty.")

    running_mean, running_stderr = _running_mean_and_stderr(x)
    n = int(x.size)

    # Downsample indices for speed/clarity (keep monotone)
    if downsample_to is not None and n > int(downsample_to) and int(downsample_to) > 50:
        idx = np.unique(np.linspace(1, n, int(downsample_to)).astype(int))
    else:
        idx = np.arange(1, n + 1, dtype=int)

    xs = idx
    ys = running_mean[idx - 1]
    se = running_stderr[idx - 1]

    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.0))

    ax.plot(xs, ys, linewidth=1.8, label="Running mean PV")

    if band_sigma and band_sigma > 0.0:
        lo = ys - float(band_sigma) * se
        hi = ys + float(band_sigma) * se
        ax.fill_between(xs, lo, hi, alpha=0.20, label=f"±{band_sigma:.0f}×running stderr")

    if pv_benchmark is not None:
        ax.axhline(float(pv_benchmark), linewidth=1.8, linestyle="--", label=str(benchmark_label))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best", frameon=True)

    return ax


def plot_stderr_scaling_vs_paths(
    *,
    points: Iterable[McConvergencePoint],
    title: str = "Standard error vs number of paths (expected ~ 1/sqrt(N))",
    xlabel: str = "Number of paths",
    ylabel: str = "Standard error of PV",
    show_reference_slope: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot stderr vs n_paths on log-log axes and optionally overlay the theoretical 1/sqrt(N) scaling.

    What to look for
    ----------------
    - On log-log axes, stderr should fall roughly on a straight line.
    - Slope should be ~ -0.5 if the variance is stable and sampling is iid.

    Parameters
    ----------
    points:
        McConvergencePoint objects (must include pv_stderr).
    show_reference_slope:
        If True, overlays a reference curve anchored at the first point with slope -0.5.
    ax:
        Optional axis.

    Returns
    -------
    Axes
    """
    pts = _as_sorted_points(points)
    n_paths = np.asarray([p.n_paths for p in pts], dtype=np.float64)
    stderr = np.asarray([p.pv_stderr for p in pts], dtype=np.float64)

    if np.any(stderr <= 0.0) or np.any(~np.isfinite(stderr)):
        raise ValueError("All pv_stderr values must be finite and > 0.")

    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.0))

    ax.plot(n_paths, stderr, marker="o", linewidth=1.8, label="Observed stderr")

    if show_reference_slope and n_paths.size >= 2:
        # Reference: c / sqrt(N), anchored at first point (N0, se0)
        n0 = float(n_paths[0])
        se0 = float(stderr[0])
        ref = se0 * np.sqrt(n0 / n_paths)
        ax.plot(n_paths, ref, linestyle="--", linewidth=1.6, label="Reference: ~ 1/sqrt(N)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.legend(loc="best", frameon=True)

    return ax
