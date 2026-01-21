from __future__ import annotations

"""
Example 04: Build a MarketDataset using an internal provider (provider → panels → factories → dataset).

What this teaches
-----------------
1) The difference between:
   - "core spine" objects (MarketId, Panel, factories, MarketDataset)
   - "integration layer" objects (providers that fetch / generate numbers)

2) A realistic ingestion flow:
      Provider -> numpy arrays -> Panels -> MarketDataset -> snapshot -> Market

3) How to keep the provider contract simple:
   - provider returns numeric arrays on deterministic grids
   - the rest of the library remains unchanged

Notes
-----
- This is a *demo provider* implemented in the example for clarity.
- In production you would move the provider into src/marketdata/providers/...
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Protocol, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.core.reporting.plots.marketdata.curves import plot_curve_df, plot_curve_zero_rate
from src.core.reporting.plots.marketdata.quotes import plot_quotes
from src.core.reporting.plots.marketdata.surfaces import plot_vol_smile_slices, plot_vol_surface, plot_vol_surface_heatmap
from src.core.reporting.plots.utils import PlotConfig, render_fig

from src.marketdata.builders.datasets import build_marketdataset
from src.marketdata.builders.panels import make_grid_vol_panel, make_quote_panel, make_time_grid, make_zero_curve_panel
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.panel import Panel
from src.marketdata.curves.factory import ZeroRateCurveFactory
from src.marketdata.surfaces.factory import GridVolFactory


# ======================================================================================
# 1) Canonical MarketIds (the “spine” keys)
# ======================================================================================

@dataclass(frozen=True, slots=True)
class ExampleIds:
    """MarketId bundle used both for dataset storage keys and snapshot lookups."""
    spot_eurusd: MarketId
    usd_curve: MarketId
    eur_curve: MarketId
    eurusd_vol: MarketId


def make_example_ids() -> ExampleIds:
    """Construct the canonical MarketIds used throughout this example."""
    return ExampleIds(
        spot_eurusd=MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD"),
        usd_curve=MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),)),
        eur_curve=MarketId(asset_class="IR", mkt_type="CURVE", name="EUR.OIS", qualifiers=(("ccy", "EUR"),)),
        eurusd_vol=MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD", qualifiers=(("cut", "NY"), ("conv", "delta25"))),
    )


# ======================================================================================
# 2) Provider contract (minimal + pragmatic)
# ======================================================================================

class MarketDataProvider(Protocol):
    """
    Minimal provider interface for this example.

    Design choice:
    - provider returns deterministic arrays on grids we define here
    - panels + factories reconstruct objects at snapshot time
    """

    def get_spot_ts(self, *, spot_id: MarketId, dates: list[str], n_scenarios: int) -> np.ndarray:
        """Return spot array shaped [T,S]."""

    def get_zero_curve_ts(
        self,
        *,
        curve_id: MarketId,
        dates: list[str],
        n_scenarios: int,
        tenors: np.ndarray,
    ) -> np.ndarray:
        """Return zero-rate array shaped [T,S,K] aligned to provided tenors."""

    def get_grid_vol_ts(
        self,
        *,
        vol_id: MarketId,
        dates: list[str],
        n_scenarios: int,
        expiries: np.ndarray,
        strikes: np.ndarray,
        spot_ts: np.ndarray,
    ) -> np.ndarray:
        """Return vol grid shaped [T,S,n_exp,n_k]."""


@dataclass(frozen=True, slots=True)
class DemoProvider:
    """
    A tiny deterministic provider.

    This simulates:
    - spot drift varies by time + scenario
    - curve levels vary by time + scenario
    - vol surface depends on spot + time + scenario

    In real life:
    - spot comes from FX spot feed
    - curves from curve bootstrapping
    - vols from market smile quotes / surfaces
    """

    def get_spot_ts(self, *, spot_id: MarketId, dates: list[str], n_scenarios: int) -> np.ndarray:
        T = len(dates)
        S = int(n_scenarios)

        # Time index used to generate smooth deterministic evolution.
        t_idx = np.arange(T, dtype=float)

        # Build [T,S] spot paths (toy deterministic drift by scenario).
        spot = np.empty((T, S), dtype=float)
        for si in range(S):
            spot[:, si] = 1.10 * (1.0 + 0.0010 * t_idx + 0.0005 * si)

        return spot

    def get_zero_curve_ts(
        self,
        *,
        curve_id: MarketId,
        dates: list[str],
        n_scenarios: int,
        tenors: np.ndarray,
    ) -> np.ndarray:
        T = len(dates)
        S = int(n_scenarios)
        ten = np.asarray(tenors, dtype=float).reshape(-1)
        K = ten.size

        # Pick a base curve depending on the curve "name".
        if curve_id.name.startswith("USD"):
            base = np.array([0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.050], dtype=float)
        else:
            base = np.array([0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.037], dtype=float)

        if base.size != K:
            raise ValueError("DemoProvider.get_zero_curve_ts: base curve length must match tenors length.")

        out = np.empty((T, S, K), dtype=float)

        # Mild time/scenario bumps so curves change across the cube.
        for ti in range(T):
            for si in range(S):
                out[ti, si, :] = base + 0.0002 * ti + 0.0005 * si

        return out

    def get_grid_vol_ts(
        self,
        *,
        vol_id: MarketId,
        dates: list[str],
        n_scenarios: int,
        expiries: np.ndarray,
        strikes: np.ndarray,
        spot_ts: np.ndarray,
    ) -> np.ndarray:
        T = len(dates)
        S = int(n_scenarios)
        exp = np.asarray(expiries, dtype=float).reshape(-1)
        k = np.asarray(strikes, dtype=float).reshape(-1)

        out = np.empty((T, S, exp.size, k.size), dtype=float)

        for ti in range(T):
            for si in range(S):
                spot = float(spot_ts[ti, si])
                log_m = np.log(k / spot)

                # “ATM” varies with time/scenario just to make the cube non-flat.
                atm = 0.12 + 0.002 * (ti / max(T - 1, 1)) + 0.002 * si

                # Fill each expiry slice with a simple skew + curvature structure.
                for i, Texp in enumerate(exp.tolist()):
                    skew = -0.06 * np.sqrt(float(Texp))
                    curv = 0.18 / (1.0 + float(Texp))
                    out[ti, si, i, :] = np.maximum(atm + skew * log_m + curv * (log_m ** 2), 1e-8)

        return out


# ======================================================================================
# 3) Provider → Panels → MarketDataset
# ======================================================================================

def build_dataset_from_provider(
    *,
    provider: MarketDataProvider,
    ids: ExampleIds,
    n_time: int,
    n_scenarios: int,
) -> Tuple[MarketDataset, ExampleIds]:
    """
    Build a MarketDataset by calling a provider for raw arrays, then converting them into Panels.

    This is the “real world” composition point:
    - providers don’t know about MarketDataset internals
    - MarketDataset doesn’t know about external data sources
    """
    # Build a deterministic time grid used to index the dataset.
    dates = make_time_grid(start="2026-01-07", n_t=int(n_time), step="D")

    # -------------------------------
    # Quotes: provider -> [T,S] -> Panel
    # -------------------------------
    spot_ts = provider.get_spot_ts(spot_id=ids.spot_eurusd, dates=dates, n_scenarios=int(n_scenarios))

    # make_quote_panel accepts either arrays or a callable; we pass the array directly.
    spot_panel = make_quote_panel(n_t=int(n_time), n_s=int(n_scenarios), values=spot_ts)
    quote_panels: Dict[MarketId, Panel] = {ids.spot_eurusd: spot_panel}

    # -------------------------------
    # Curves: provider -> [T,S,K] -> Panel([T,S,K,2]) + factory
    # -------------------------------
    tenors = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)

    usd_zero_ts = provider.get_zero_curve_ts(curve_id=ids.usd_curve, dates=dates, n_scenarios=int(n_scenarios), tenors=tenors)
    eur_zero_ts = provider.get_zero_curve_ts(curve_id=ids.eur_curve, dates=dates, n_scenarios=int(n_scenarios), tenors=tenors)

    # Convert [T,S,K] into a callable (ti,si) -> [K] so make_zero_curve_panel can build [T,S,K,2].
    usd_curve_panel = make_zero_curve_panel(
        dates=dates,
        n_s=int(n_scenarios),
        tenors=tenors,
        zero_rates=lambda ti, si: usd_zero_ts[int(ti), int(si), :],
    )
    eur_curve_panel = make_zero_curve_panel(
        dates=dates,
        n_s=int(n_scenarios),
        tenors=tenors,
        zero_rates=lambda ti, si: eur_zero_ts[int(ti), int(si), :],
    )

    curve_param_panels: Dict[MarketId, Panel] = {
        ids.usd_curve: usd_curve_panel,
        ids.eur_curve: eur_curve_panel,
    }
    curve_factories = {
        ids.usd_curve: ZeroRateCurveFactory(extrapolation="flat"),
        ids.eur_curve: ZeroRateCurveFactory(extrapolation="flat"),
    }

    # -------------------------------
    # Vols: provider -> [T,S,n_exp,n_k] -> Panel + factory
    # -------------------------------
    expiries = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    strikes = 1.10 * np.exp(np.linspace(-0.25, 0.25, 31, dtype=float))

    vol_ts = provider.get_grid_vol_ts(
        vol_id=ids.eurusd_vol,
        dates=dates,
        n_scenarios=int(n_scenarios),
        expiries=expiries,
        strikes=strikes,
        spot_ts=spot_ts,
    )

    # Here we pass a generator that returns exactly the provider grid for (ti,si).
    # make_grid_vol_panel will store the resulting panel as [T,S,n_exp,n_k].
    vol_panel = make_grid_vol_panel(
        dates=dates,
        n_s=int(n_scenarios),
        spot_panel=spot_panel,
        expiries=expiries,
        strikes=strikes,
        vol_grid=lambda spot, ti, si, exp, k: vol_ts[int(ti), int(si), :, :],
    )

    vol_param_panels: Dict[MarketId, Panel] = {ids.eurusd_vol: vol_panel}
    vol_factories = {ids.eurusd_vol: GridVolFactory(expiries=expiries, strikes=strikes, extrapolation="flat")}

    # -------------------------------
    # Assemble dataset (one consistent wiring path)
    # -------------------------------
    dataset = build_marketdataset(
        dates=dates,
        n_scenarios=int(n_scenarios),
        quote_panels=quote_panels,
        curve_param_panels=curve_param_panels,
        curve_factories=curve_factories,
        vol_param_panels=vol_param_panels,
        vol_factories=vol_factories,
        meta={"source": "demo_provider", "purpose": "04_market_dataset_from_provider"},
        validate=True,
    )

    return dataset, ids


# ======================================================================================
# 4) Snapshot + plotting (same style as Example 02)
# ======================================================================================

def print_snapshot_summary(*, market: Market, ids: ExampleIds, time_idx: int, scenario_idx: int) -> None:
    """Print a quick numeric sanity check for one snapshot."""
    spot = float(market.quote(ids.spot_eurusd))
    usd_curve = market.curve(ids.usd_curve)
    eur_curve = market.curve(ids.eur_curve)
    vol = market.vol_surface(ids.eurusd_vol)

    print("\n" + "=" * 95)
    print(f"SNAPSHOT  t={time_idx}  s={scenario_idx}  asof={market.asof}")
    print("=" * 95)
    print(f"Spot(EURUSD): {spot:.6f}")
    print(f"USD curve: df(1y)={usd_curve.df(1.0):.8f}  r(1y)={usd_curve.zero_rate(1.0):.6f}")
    print(f"EUR curve: df(1y)={eur_curve.df(1.0):.8f}  r(1y)={eur_curve.zero_rate(1.0):.6f}")
    print(f"EURUSD vol: sigma(1y, ATM~spot)={vol.implied_vol(1.0, spot):.6f}")


def plot_snapshot_objects(*, market: Market, ids: ExampleIds, plot_cfg: PlotConfig, time_idx: int, scenario_idx: int) -> None:
    """Plot the snapshot objects (quotes + curves + vols) with optional saving."""
    plt.ioff()

    figures = [
        ("01_quotes.png", plot_quotes(market.quotes, title=f"Quotes (t={time_idx}, s={scenario_idx})")),
        ("02_usd_df.png", plot_curve_df(market.curve(ids.usd_curve), title="USD curve: Discount Factor DF(t)")),
        ("03_usd_zero.png", plot_curve_zero_rate(market.curve(ids.usd_curve), title="USD curve: Zero Rate r(t)")),
        ("04_eur_df.png", plot_curve_df(market.curve(ids.eur_curve), title="EUR curve: Discount Factor DF(t)")),
        ("05_eur_zero.png", plot_curve_zero_rate(market.curve(ids.eur_curve), title="EUR curve: Zero Rate r(t)")),
    ]

    vol = market.vol_surface(ids.eurusd_vol)
    figures.extend([
        ("06_vol_heatmap.png", plot_vol_surface_heatmap(vol, title="EURUSD vol (heatmap)")),
        ("07_vol_smiles.png", plot_vol_smile_slices(vol, title="EURUSD smile slices σ(T,K)")),
        ("08_vol_surface_3d.png", plot_vol_surface(vol, title="EURUSD vol (3D)")),
    ])

    for filename, fig in figures:
        render_fig(fig, cfg=plot_cfg, filename=filename)


def main() -> None:
    """Run the provider → dataset → snapshot workflow."""
    plot_cfg = PlotConfig(
        show=True,
        save=False,
        out_dir=Path("outputs/marketdata/04_market_dataset_from_provider"),
        block=True,
        close=False,
        dpi=160,
    )

    ids = make_example_ids()
    provider = DemoProvider()

    dataset, ids = build_dataset_from_provider(provider=provider, ids=ids, n_time=6, n_scenarios=3)

    time_idx = 2
    scenario_idx = 1

    market = dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)

    print_snapshot_summary(market=market, ids=ids, time_idx=time_idx, scenario_idx=scenario_idx)
    plot_snapshot_objects(market=market, ids=ids, plot_cfg=plot_cfg, time_idx=time_idx, scenario_idx=scenario_idx)


if __name__ == "__main__":
    main()