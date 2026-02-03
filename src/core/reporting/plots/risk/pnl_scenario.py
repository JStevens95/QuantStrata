"""
PnL by scenario and attribution bar plots for risk reports.

Consumes ScenarioReport and AttributionReport; uses report style for
publication-quality output.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from src.core.reporting.plots.style import apply_report_style, get_report_figsize
from src.risk.attribution.report import AttributionReport
from src.risk.reporting.scenario_report import ScenarioReport


def plot_pnl_by_scenario(
    scenario_report: ScenarioReport,
    *,
    title: Optional[str] = None,
    exclude_base: bool = True,
) -> plt.Figure:
    """
    Bar chart of PnL per scenario (excluding base by default).

    Parameters
    ----------
    scenario_report : ScenarioReport
        Output from scenario runner or build_scenario_report().
    title : str, optional
        Figure title.
    exclude_base : bool
        If True, omit the base scenario (PnL=0) from the chart.

    Returns
    -------
    plt.Figure
    """
    rows = scenario_report.rows
    if exclude_base:
        rows = [r for r in rows if r.scenario != scenario_report.base_scenario]
    if not rows:
        # Empty: still return a figure with no bars
        labels = []
        pnl = np.array([], dtype=float)
    else:
        labels = [r.scenario for r in rows]
        pnl = np.array([r.pnl for r in rows], dtype=float)

    fig = plt.figure(figsize=get_report_figsize())
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    colors = np.where(pnl >= 0, "C0", "C3")
    ax.bar(x, pnl, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title or "PnL by scenario")
    ax.set_ylabel("PnL")
    ax.axhline(0.0, linestyle="-", linewidth=0.8, color="gray")
    apply_report_style(ax)
    fig.tight_layout()
    return fig


def plot_attribution_bars(
    attribution_report: AttributionReport,
    *,
    scenario: Optional[str] = None,
    title: Optional[str] = None,
    max_contributions: int = 12,
) -> plt.Figure:
    """
    Bar chart of factor contributions for one scenario (or first non-base).

    Parameters
    ----------
    attribution_report : AttributionReport
        Output from attribute_portfolio_scenarios().
    scenario : str, optional
        Which scenario to plot. If None, use the first non-base scenario.
    title : str, optional
        Figure title.
    max_contributions : int
        Max number of contribution bars to show (sorted by abs value).

    Returns
    -------
    plt.Figure
    """
    rows = [r for r in attribution_report.rows if r.scenario != attribution_report.base_scenario_name]
    if not rows:
        labels = []
        contrib = np.array([], dtype=float)
        row_label = attribution_report.base_scenario_name
    else:
        row = rows[0] if scenario is None else next((r for r in rows if r.scenario == scenario), rows[0])
        row_label = row.scenario
        contrib_dict = dict(row.contributions)
        # Sort by absolute contribution, take top max_contributions
        sorted_items = sorted(contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:max_contributions]
        labels = [k for k, _ in sorted_items]
        contrib = np.array([v for _, v in sorted_items], dtype=float)

    fig = plt.figure(figsize=get_report_figsize())
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    colors = np.where(contrib >= 0, "C0", "C3")
    ax.bar(x, contrib, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title or f"PnL attribution: {row_label}")
    ax.set_ylabel("Contribution")
    ax.axhline(0.0, linestyle="-", linewidth=0.8, color="gray")
    apply_report_style(ax)
    fig.tight_layout()
    return fig


__all__ = ["plot_attribution_bars", "plot_pnl_by_scenario"]
