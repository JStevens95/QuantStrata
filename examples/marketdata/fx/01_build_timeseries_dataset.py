from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import TimeseriesRequest, Universe
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.market import Market

from src.marketdata.providers.factory import SyntheticProviderSpec, build_provider

from src.core.reporting.plots.marketdata.quotes import plot_quotes
from src.core.reporting.plots.marketdata.curves import (
    plot_curve_df, plot_curve_zero_rate, plot_curve_forward_rate,
)
from src.core.reporting.plots.marketdata.surfaces import (
    plot_vol_surface_heatmap, plot_vol_smile_slices, plot_vol_surface,
)
from src.core.reporting.plots.marketdata.scenarios import (
    plot_spot_comparison, plot_curve_df_comparison, plot_vol_comparison,
)
from src.core.reporting.plots.marketdata.timeseries import (
    FanSpec, flatten_log_returns_all_scenarios, plot_log_return_timeseries, plot_return_correlation_heatmap,
    plot_spot_fan_chart,
)

# Synthetic config + specs for professional multi-product realism.
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.specs import SpotGbmSpec, VolGridSmileSpec


# --------------------------------------------------------------------------------------
# Example goal
# --------------------------------------------------------------------------------------
# This example demonstrates an end-to-end "market pack" workflow that is common on a
# front-office quant desk:
#
#  1) Build a MarketDataset for a date range and scenario count (timeseries + scenarios)
#  2) Plot timeseries diagnostics (spots + returns + corr)
#  3) Slice a pricing-ready Market snapshot (as-of a date & scenario)
#  4) Plot the snapshot objects (quotes / curves / vols)
#  5) Compare scenario snapshots using your scenario comparison plots
#
# Key design principle enforced by your library:
#   - Everything downstream (pricing, risk) consumes *Market*, not provider internals.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExampleConfig:
    """
    Central configuration for the example.

    Keeping everything in one place makes the example explicit and easy to modify.
    """
    start: str = "2026-01-01"
    end: str = "2026-02-01"
    freq: str = "D"
    n_scenarios: int = 50

    # Snapshot selection (for “desk pack” plots).
    snapshot_time_idx: int = -1  # last date in the dataset
    snapshot_scenario_idx: int = 0  # base scenario

    # Output directory for plots.
    out_dir_name: str = "outputs/01_build_timeseries_dataset"


def _fx_vol_id(pair: str, *, cut: str, convention: str, dom: str, foreign: str) -> MarketId:
    """
    Construct an FX VOL MarketId with explicit dom/for qualifiers.

    Important:
    - Your FX VOL generator only uses dom/for curves if these qualifiers are present.
    - Your dependency closure copies qualifiers to the SPOT dependency, so we keep
      qualifiers consistent and explicit.
    """
    # We use a tuple-of-pairs qualifiers representation for determinism and hashability.
    qualifiers: Tuple[Tuple[str, str], ...] = (
        ("cut", str(cut)),
        ("convention", str(convention)),
        ("dom", str(dom).upper()),
        ("for", str(foreign).upper()),
    )

    return MarketId(asset_class="FX", mkt_type="VOL", name=str(pair).upper(), qualifiers=qualifiers)


def _fx_spot_id_from_vol(vol_id: MarketId) -> MarketId:
    """
    Build the corresponding SPOT MarketId that your FX VOL generator requires.

    This mirrors your generator's requirements logic:
      spot_id = MarketId(asset_class=..., mkt_type="SPOT", name=..., qualifiers=vol_id.qualifiers)
    """
    return MarketId(
        asset_class=vol_id.asset_class,
        mkt_type="SPOT",
        name=vol_id.name,
        qualifiers=vol_id.qualifiers,
    )


def _ir_curve_id(ccy: str) -> MarketId:
    """
    Construct the canonical IR curve id used by your FX VOL generator.

    This matches your internal helper _default_ir_curve_id(ccy):
      IR.CURVE.<CCY>.OIS|ccy=<CCY>
    """
    c = str(ccy).strip().upper()
    return MarketId(asset_class="IR", mkt_type="CURVE", name=f"{c}.OIS", qualifiers=(("ccy", c),))


def _resolve_index(idx: int, *, size: int, name: str) -> int:
    """
    Resolve Python-style negative indices into [0, size-1], and validate bounds.

    This keeps the example user-friendly while preserving strict core-library behavior.
    """
    i = int(idx)
    n = int(size)

    if n <= 0:
        raise ValueError(f"Cannot resolve {name} index: size={n}.")

    # Convert negative indexing (e.g. -1 -> last element).
    if i < 0:
        i = n + i

    # Validate bounds explicitly.
    if i < 0 or i >= n:
        raise IndexError(f"{name}_idx out of range: {idx} -> {i} for {name.upper()}={n}.")

    return i


def _build_provider_with_realistic_overrides(*, vol_ids: Sequence[MarketId]) -> object:
    """
    Build a SyntheticProvider with per-product overrides so the example looks “market-like”.

    Why overrides matter
    --------------------
    Your default specs are tuned for EURUSD-like scales (spot ~1.10, strikes ~0.9..1.2).
    USDJPY requires a different scale (spot ~110, strikes ~90..130).
    """
    # Build the overrides keyed by the exact MarketId instances we will generate.
    spot_overrides: Dict[MarketId, SpotGbmSpec] = {}
    vol_overrides: Dict[MarketId, VolGridSmileSpec] = {}

    # We derive the *exact* SPOT MarketId that the VOL dependency closure will generate.
    for vol_id in vol_ids:
        spot_id = _fx_spot_id_from_vol(vol_id)

        # Pair-specific tuning. Keep it explicit for readability.
        pair = str(vol_id.name).upper()

        if pair == "USDJPY":
            # USDJPY: spot around 110 and strikes around 90..130.
            spot_overrides[spot_id] = SpotGbmSpec(initial_level=110.0, vol=0.10)
            vol_overrides[vol_id] = VolGridSmileSpec(
                expiries=np.array([0.25, 0.50, 1.00, 2.00], dtype=float),
                strikes=np.array([90.0, 100.0, 110.0, 120.0, 130.0], dtype=float),
                atm_vol=0.11,
                skew=-0.10,
                smile=0.18,
                term=0.08,
                noise_scale=0.0015,
            )
        else:
            # EURUSD / GBPUSD: leave strikes in the ~0.9..1.2 range.
            spot_overrides[spot_id] = SpotGbmSpec(initial_level=1.10, vol=0.12)
            vol_overrides[vol_id] = VolGridSmileSpec(
                expiries=np.array([0.25, 0.50, 1.00, 2.00], dtype=float),
                strikes=np.array([0.90, 1.00, 1.10, 1.20], dtype=float),
                atm_vol=0.12,
                skew=-0.15,
                smile=0.20,
                term=0.10,
                noise_scale=0.0020,
            )

    # Assemble the synthetic provider config with overrides.
    cfg = SyntheticProviderConfig(
        spot_overrides=spot_overrides,
        vol_overrides=vol_overrides,
        curve_method="zeros",  # stable default; bootstrap path is plumbed but intentionally conservative today
    )

    # Build provider via your factory so examples remain provider-agnostic.
    provider = build_provider(
        SyntheticProviderSpec(
            seed=7,              # deterministic example run
            config=cfg,
            name="SyntheticProvider",
        )
    )
    return provider


def _print_dataset_summary(*, dataset: MarketDataset, requested_ids: Sequence[MarketId]) -> None:
    """
    Print a concise, front-office friendly dataset summary.
    """
    print("\n=== FX MarketDataset (Multi-Product: SPOT + CURVE + VOL) ===")
    print(f"Provider:         {dataset.meta.get('provider', 'UNKNOWN') if dataset.meta else 'UNKNOWN'}")
    print(f"Requested IDs:    {[m.key() for m in requested_ids]}")
    print(
        f"Date range:       {dataset.dates[0]} -> {dataset.dates[-1]} "
        f"(freq={dataset.meta.get('freq') if dataset.meta else '?'})"
    )
    print(f"n_time:           {len(dataset.dates)}")
    print(f"n_scenarios:      {dataset.n_scenarios}")

    meta = dataset.meta or {}
    print("\n--- Dataset meta (selected) ---")
    for k in ["seed", "requested_scenarios", "n_time", "n_scenarios", "n_requested_ids", "n_closed_ids"]:
        if k in meta:
            print(f"{k:>18}: {meta[k]}")

    print("\n--- Panels present ---")
    for mid in sorted(dataset.panels.keys(), key=lambda m: m.key()):
        print(f" - {mid.key()}")

    print("\n--- Curves present ---")
    for mid in sorted(dataset.curve_params.keys(), key=lambda m: m.key()):
        print(f" - {mid.key()}")

    print("\n--- Vol surfaces present ---")
    for mid in sorted(dataset.vol_params.keys(), key=lambda m: m.key()):
        print(f" - {mid.key()}")


def _extract_spot_panel_or_raise(*, dataset: MarketDataset, spot_id: MarketId) -> np.ndarray:
    """
    Extract a spot panel [T,S] from the MarketDataset.

    We read directly from dataset.panels (not from Market.quote()) because this is
    a timeseries example, and panels are the canonical storage for [T,S] data.
    """
    try:
        panel = dataset.panels[spot_id]
    except KeyError as exc:
        raise KeyError(f"Spot panel not found for {spot_id.key()}") from exc

    arr = np.asarray(panel.data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected spot panel [T,S] for {spot_id.key()}, got shape={arr.shape}.")
    return arr


def _emit_figure(*, fig: plt.Figure, save_files: bool, path: Path) -> None:
    """
    Either save a matplotlib figure to disk (save_files=True) or keep it open for plt.show() (save_files=False).

    Why this helper exists
    ----------------------
    - Keeps the example ergonomic for interactive use (no filesystem writes by default).
    - Keeps CI / batch runs deterministic when saving is enabled.
    """
    if not bool(save_files):
        # Do nothing: keep the figure open so the user can interact with it via plt.show().
        return

    # When saving, ensure output folder exists and close figures to avoid memory buildup.
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(save_files: bool = False) -> None:
    """
    Run the example end-to-end.

    Parameters
    ----------
    save_files:
        - False (default): display figures interactively (no file outputs)
        - True           : save figures to outputs/... and close figures
    """
    # Basic logging setup (kept minimal; orchestrator examples will use full run logger).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = ExampleConfig()

    # Resolve an absolute output directory relative to this script file.
    # IMPORTANT:
    # - We always compute out_dir (for consistent paths)
    # - We only create it on disk if save_files=True
    out_dir = (Path(__file__).resolve().parent / cfg.out_dir_name).resolve()
    if save_files:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Define "desk pack" market ids (VOL drives dependency closure)
    # ------------------------------------------------------------------
    # We request vol surfaces with explicit dom/for qualifiers.
    # The synthetic engine will automatically include SPOT + required IR curves.
    vol_ids: List[MarketId] = [
        _fx_vol_id("EURUSD", cut="NY", convention="delta25", dom="USD", foreign="EUR"),
        _fx_vol_id("GBPUSD", cut="NY", convention="delta25", dom="USD", foreign="GBP"),
        _fx_vol_id("USDJPY", cut="TK", convention="delta25", dom="JPY", foreign="USD"),
    ]

    # For clarity, we also define the corresponding SPOT ids (so we can plot them).
    spot_ids = [_fx_spot_id_from_vol(v) for v in vol_ids]

    # And the curves we expect to exist due to dom/for qualifiers.
    curve_ids: List[MarketId] = [
        _ir_curve_id("USD"),
        _ir_curve_id("EUR"),
        _ir_curve_id("GBP"),
        _ir_curve_id("JPY"),
    ]

    # Build the universe for the request.
    # Note: we include vols (required), and curves (optional) for explicitness.
    requested_ids: List[MarketId] = list(vol_ids) + list(curve_ids)
    universe = Universe(ids=requested_ids)

    # ------------------------------------------------------------------
    # 2) Build provider + request dataset
    # ------------------------------------------------------------------
    provider = _build_provider_with_realistic_overrides(vol_ids=vol_ids)

    request = TimeseriesRequest(
        start=cfg.start,
        end=cfg.end,
        freq=cfg.freq,
        universe=universe,
        scenarios=int(cfg.n_scenarios),
    )

    dataset = provider.get_timeseries(request)
    _print_dataset_summary(dataset=dataset, requested_ids=requested_ids)

    # ------------------------------------------------------------------
    # 3) Timeseries plots (from dataset.panels)
    # ------------------------------------------------------------------
    # We build:
    #  - fan charts per pair
    #  - median log-return timeseries per pair
    #  - return correlation heatmap across pairs
    returns_by_pair: Dict[str, np.ndarray] = {}

    for spot_id in spot_ids:
        pair = str(spot_id.name).upper()
        spot_paths = _extract_spot_panel_or_raise(dataset=dataset, spot_id=spot_id)

        # Spot fan chart.
        fig_fan = plot_spot_fan_chart(
            dates=dataset.dates,
            spot_paths=spot_paths,
            title=f"{pair} spot fan chart ({cfg.start} -> {cfg.end}, S={dataset.n_scenarios})",
            fan=FanSpec(q_low=0.05, q_mid=0.50, q_high=0.95, max_scenario_lines=8),
        )
        _emit_figure(fig=fig_fan, save_files=save_files, path=out_dir / f"timeseries_spot_fan_{pair}.png")

        # Median log returns.
        fig_ret = plot_log_return_timeseries(
            dates=dataset.dates,
            spot_paths=spot_paths,
            title=f"{pair} median log returns (scenario median)",
        )
        _emit_figure(fig=fig_ret, save_files=save_files, path=out_dir / f"timeseries_log_returns_{pair}.png")

        # Flatten returns across all scenarios for correlation diagnostics.
        returns_by_pair[pair] = flatten_log_returns_all_scenarios(spot_paths)

    # Correlation heatmap across instruments.
    fig_corr = plot_return_correlation_heatmap(
        returns_by_label=returns_by_pair,
        title="FX spot log-return correlation (all scenarios flattened)",
    )
    _emit_figure(fig=fig_corr, save_files=save_files, path=out_dir / "timeseries_return_corr_heatmap.png")

    # ------------------------------------------------------------------
    # 4) Snapshot a pricing-ready Market and plot desk objects
    # ------------------------------------------------------------------
    # This is the crucial “front-office” step:
    # - Market is what pricers consume (quotes/curves/vols ready to use)
    time_idx = _resolve_index(cfg.snapshot_time_idx, size=len(dataset.dates), name="time")
    scen_idx = _resolve_index(cfg.snapshot_scenario_idx, size=int(dataset.n_scenarios), name="scenario")
    market: Market = dataset.snapshot(time_idx=time_idx, scenario_idx=scen_idx)

    # 4a) Quotes bar chart (SPOT quotes at this snapshot).
    # IMPORTANT: plot_quotes expects Mapping[MarketId, Quote], so we slice market.quotes.
    spot_quotes = {mid: market.quotes[mid] for mid in spot_ids if mid in market.quotes}
    fig_q = plot_quotes(spot_quotes, title=f"FX Spots @ {market.asof} (scenario={scen_idx})")
    _emit_figure(fig=fig_q, save_files=save_files, path=out_dir / "snapshot_quotes_spots.png")

    # 4b) Curves: DF(t), zero rate, forward rate.
    # We plot a couple curves to keep it readable; add more if desired.
    for curve_id in curve_ids:
        if curve_id not in market.curves:
            # Be explicit but do not fail the entire example if a curve is absent.
            logging.warning("Curve not present in snapshot: %s", curve_id.key())
            continue

        curve = market.curve(curve_id)
        ccy = str(curve_id.name).split(".")[0]

        fig_df = plot_curve_df(curve, title=f"{ccy} DF(t) @ {market.asof}", t_max=10.0)
        _emit_figure(fig=fig_df, save_files=save_files, path=out_dir / f"snapshot_curve_df_{ccy}.png")

        fig_zero = plot_curve_zero_rate(curve, title=f"{ccy} zero rate r(t) @ {market.asof}", t_max=10.0)
        _emit_figure(fig=fig_zero, save_files=save_files, path=out_dir / f"snapshot_curve_zero_{ccy}.png")

        fig_fwd = plot_curve_forward_rate(curve, t1=0.5, t2_max=10.0, title=f"{ccy} forward f(0.5,t) @ {market.asof}")
        _emit_figure(fig=fig_fwd, save_files=save_files, path=out_dir / f"snapshot_curve_fwd_{ccy}.png")

    # 4c) Vol surfaces: heatmap + smile slices + optional 3D.
    # We plot for EURUSD and GBPUSD (USDJPY also works because we overrode strike scale).
    for vol_id in vol_ids:
        if vol_id not in market.vols:
            logging.warning("Vol surface not present in snapshot: %s", vol_id.key())
            continue

        pair = str(vol_id.name).upper()
        surface = market.vol_surface(vol_id)

        fig_hm = plot_vol_surface_heatmap(surface, title=f"{pair} vol heatmap @ {market.asof} (scen={scen_idx})")
        _emit_figure(fig=fig_hm, save_files=save_files, path=out_dir / f"snapshot_vol_heatmap_{pair}.png")

        fig_smile = plot_vol_smile_slices(surface, title=f"{pair} smile slices @ {market.asof} (scen={scen_idx})")
        _emit_figure(fig=fig_smile, save_files=save_files, path=out_dir / f"snapshot_vol_smiles_{pair}.png")

        # 3D surface is optional but useful for presentations / desk reports.
        fig_3d = plot_vol_surface(surface, title=f"{pair} 3D vol surface @ {market.asof} (scen={scen_idx})")
        _emit_figure(fig=fig_3d, save_files=save_files, path=out_dir / f"snapshot_vol_3d_{pair}.png")

    # ------------------------------------------------------------------
    # 5) Scenario comparisons using snapshot markets (Market satisfies MarketView)
    # ------------------------------------------------------------------
    # We compare scenario snapshots (not “shock scenarios”) to show dispersion.
    # This is still very front-office relevant: scenario 0 vs other scenarios
    # demonstrates how quotes/curves/vols change in your dataset.
    base = dataset.snapshot(time_idx=time_idx, scenario_idx=0)

    shocked = {
        "SCEN_1": dataset.snapshot(time_idx=time_idx, scenario_idx=1),
        "SCEN_10": dataset.snapshot(time_idx=time_idx, scenario_idx=10),
    }

    # 5a) Spot comparison bars (choose a representative spot).
    spot_id0 = spot_ids[0]
    fig_s, _ = plot_spot_comparison(base=base, shocked=shocked, spot_id=spot_id0)
    _emit_figure(fig=fig_s, save_files=save_files, path=out_dir / "scenario_compare_spot.png")

    # 5b) Curve DF comparison (choose a representative curve).
    # We compare DF(t) across scenario snapshots at a chosen time grid.
    curve_id0 = _ir_curve_id("USD")
    times = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)

    if curve_id0 in base.curves:
        fig_c, _ = plot_curve_df_comparison(base=base, shocked=shocked, curve_id=curve_id0, times=times)
        _emit_figure(fig=fig_c, save_files=save_files, path=out_dir / "scenario_compare_curve_df.png")
    else:
        logging.warning("Skipping curve scenario comparison: base curve not present for %s", curve_id0.key())

    # 5c) Vol comparison: compare vol smile at a chosen expiry across scenarios.
    vol_id0 = vol_ids[0]
    if vol_id0 in base.vols:
        # Choose strikes consistent with the surface grid (keep it simple for the example).
        # For EURUSD-like pairs, strikes are ~0.9..1.2 in our overrides.
        strikes = np.array([0.90, 1.00, 1.10, 1.20], dtype=float)
        expiry = 1.0

        fig_v, _ = plot_vol_comparison(base=base, shocked=shocked, vol_id=vol_id0, expiry=expiry, strikes=strikes)
        _emit_figure(fig=fig_v, save_files=save_files, path=out_dir / "scenario_compare_vol.png")
    else:
        logging.warning("Skipping vol scenario comparison: base vol not present for %s", vol_id0.key())

    # ------------------------------------------------------------------
    # 6) Final output behavior: save vs display
    # ------------------------------------------------------------------
    if save_files:
        print(f"\nAll plots saved to:\n  {out_dir}\n")
    else:
        # Display all figures that were created (we intentionally kept them open).
        plt.show()


if __name__ == "__main__":
    # Default behavior: display plots (no filesystem writes).
    # Set to True if you want a saved "desk pack" output folder.
    main(save_files=False)