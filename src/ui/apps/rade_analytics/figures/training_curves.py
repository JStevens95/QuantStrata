"""Training-curves chart for the Cluster Deep-Dive sub-tab (Phase E.4).

Consumes the DataFrame returned by
:meth:`~src.ui.apps.rade_analytics.data.backend.RadeBackend.training_curves_df`
— one row per epoch, columns = ``epoch`` + any per-epoch series the
trainer emitted (``train_loss``, ``val_loss``, ``mae``, …).

Render behaviour
----------------
* ``train_loss`` is **always** plotted — it is the canonical "what did
  the model learn" trace and the chart is meaningless without it.
* Additional metrics selected via the chip row are each drawn as a
  secondary line.  ``val_*`` counterparts are auto-paired in the same
  hue (dashed) as their ``train_*`` sibling, so "train + val of mae"
  reads as one colour family instead of two unrelated lines.
* Palette pickeffs through :func:`color_for_index` so chip order in
  the UI matches trace order in the legend.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

from ._theme import color_for_index, empty_figure, rade_layout, rgba


# Dashed trace for val_* series; plain line for everything else.  Kept
# module-local so it's trivial to tweak later (e.g. dotted for test_*).
_VAL_DASH = "dash"


def _is_val_pair(primary: str, candidate: str) -> bool:
    """True when ``candidate`` is the ``val_*`` sibling of ``primary``.

    Matches both ``val_loss`` (paired with ``train_loss`` / ``loss``)
    and ``val_<metric>`` (paired with ``<metric>``).
    """
    if primary in ("train_loss", "loss") and candidate == "val_loss":
        return True
    if not primary.startswith("val_") and candidate == f"val_{primary}":
        return True
    return False


def training_curves_chart(
    df: Optional[pd.DataFrame],
    *,
    selected_metrics: Optional[Sequence[str]] = None,
    available_metrics: Optional[Iterable[str]] = None,
    uirevision_key: Optional[str] = None,
) -> go.Figure:
    """Render training curves, always including ``train_loss``.

    Parameters
    ----------
    df
        DataFrame from ``training_curves_df`` (one row per epoch).
        ``None`` or empty returns an empty-state figure.
    selected_metrics
        Optional metric names to overlay alongside ``train_loss``.
        Values that are not columns of ``df`` are silently dropped.
        ``val_*`` counterparts of selected metrics are auto-added as
        dashed overlays in the same hue.
    available_metrics
        Optional pre-computed list of available metric names (from
        ``df.attrs["metrics"]``).  Only used to validate
        ``selected_metrics`` — passing it lets callers share one
        look-up across KPIs + chart + chip row without reading
        ``df.columns`` twice.
    uirevision_key
        Optional Plotly ``uirevision`` value.  Page Contract §6 —
        callers on the Cluster Deep-Dive page typically pass
        ``cluster_id`` so toggling overlay chips preserves the user's
        zoom / pan / legend toggles, while switching to a different
        cluster resets UI state to the new data domain.
    """
    if df is None or df.empty or "epoch" not in df.columns or "train_loss" not in df.columns:
        return empty_figure("Training curves unavailable for this cluster.")

    epochs = df["epoch"].tolist()
    validated = _validated_selection(df, selected_metrics, available_metrics)

    fig = go.Figure()

    # Primary trace — always train_loss.  Uses palette[0] so the colour
    # stays stable no matter what secondary metric is also selected.
    _add_primary(fig, epochs, df["train_loss"].tolist(), color_for_index(0))

    # val_loss is paired with train_loss (same hue, dashed) when it is
    # in the user's chip selection — always shown alongside its sibling
    # rather than using a separate palette slot.
    val_loss_selected = "val_loss" in validated and "val_loss" in df.columns
    if val_loss_selected:
        _add_pair(
            fig, epochs, df["val_loss"].tolist(), color_for_index(0), "val_loss",
        )

    # Secondary metrics — pair train/val per colour slot.
    slot = 1
    for metric in validated:
        if metric == "val_loss":
            # Already rendered alongside train_loss above.
            continue
        hue = color_for_index(slot)
        _add_primary(fig, epochs, df[metric].tolist(), hue, name=metric)
        val_name = f"val_{metric}" if not metric.startswith("val_") else None
        if val_name and val_name in df.columns:
            _add_pair(fig, epochs, df[val_name].tolist(), hue, val_name)
        slot += 1

    fig.update_layout(
        **rade_layout(
            show_legend=True,
            hovermode="x unified",
            xaxis={"title": "Epoch"},
            yaxis={"title": "Value"},
        )
    )
    if uirevision_key is not None:
        fig.update_layout(uirevision=uirevision_key)
    return fig


# ── internal helpers ───────────────────────────────────────────────

def _validated_selection(
    df: pd.DataFrame,
    selected: Optional[Sequence[str]],
    available: Optional[Iterable[str]],
) -> List[str]:
    """Drop unknown / duplicate / ``train_loss`` entries from selection.

    ``train_loss`` is always drawn so we strip it from the user
    selection to avoid a duplicate primary trace; every other name
    must exist as a column in ``df``.
    """
    if not selected:
        return []
    cols = set(df.columns)
    avail = set(available) if available is not None else cols
    out: List[str] = []
    seen: set[str] = set()
    for m in selected:
        if not isinstance(m, str) or not m:
            continue
        if m == "train_loss":
            continue
        if m not in cols or m not in avail:
            continue
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def _add_primary(
    fig: go.Figure,
    x: Sequence[int],
    y: Sequence[float],
    color: str,
    *,
    name: str = "train_loss",
) -> None:
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name=name,
        line={"color": color, "width": 2.2},
        hovertemplate=f"{name}: %{{y:.6f}}<extra>epoch %{{x}}</extra>",
    ))


def _add_pair(
    fig: go.Figure,
    x: Sequence[int],
    y: Sequence[float],
    color: str,
    name: str,
) -> None:
    """Add a ``val_*`` trace in the same hue but dashed + translucent."""
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name=name,
        line={"color": rgba(color, 0.75), "width": 1.6, "dash": _VAL_DASH},
        hovertemplate=f"{name}: %{{y:.6f}}<extra>epoch %{{x}}</extra>",
    ))


__all__ = ["training_curves_chart"]
