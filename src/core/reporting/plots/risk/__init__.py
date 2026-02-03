"""Risk and analytics plot utilities: Greeks surfaces, PnL by scenario."""

from src.core.reporting.plots.risk.greeks_surface import (
    greek_grid_from_sensitivities,
    plot_greeks_surface,
)
from src.core.reporting.plots.risk.pnl_scenario import (
    plot_attribution_bars,
    plot_pnl_by_scenario,
)

__all__ = [
    "greek_grid_from_sensitivities",
    "plot_attribution_bars",
    "plot_greeks_surface",
    "plot_pnl_by_scenario",
]
