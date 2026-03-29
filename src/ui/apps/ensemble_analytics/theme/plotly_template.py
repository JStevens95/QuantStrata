"""
Custom Plotly template extending ``plotly_dark``.

Ensures all figures match the dashboard's dark palette without
per-figure styling.  Registered as ``ensemble_dark`` in ``app.py``.
"""
import plotly.graph_objects as go
import plotly.io as pio

from src.ui.apps.ensemble_analytics.theme.colors import (
    BG_CARD,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    CHART_COLORS,
)

_base = pio.templates["plotly_dark"]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
            size=12,
            color=TEXT_PRIMARY,
        ),
        title=dict(font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis=dict(
            gridcolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            titlefont=dict(color=TEXT_SECONDARY, size=12),
        ),
        yaxis=dict(
            gridcolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            titlefont=dict(color=TEXT_SECONDARY, size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY, size=11),
        ),
        colorway=CHART_COLORS,
        margin=dict(l=50, r=20, t=40, b=40),
    ),
    data=_base.data,
)
