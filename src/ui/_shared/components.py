"""
Reusable Dash components for QuantStrata apps.

Use these to keep forms, result cards, and controls consistent across apps.
"""

from __future__ import annotations

try:
    from dash import dcc, html
except ImportError as e:
    raise ImportError("Dash is required. Install with: pip install dash") from e

from src.ui._shared.styles import FORM_STYLES, RESULT_STYLES


def input_row(
    id: str,
    label: str,
    *,
    type: str = "number",
    value: float | int | str = 0,
    min: float | None = None,
    max: float | None = None,
    step: float | None = None,
    placeholder: str | None = None,
) -> html.Div:
    """Single labeled input (number or text)."""
    kwargs = {"id": id, "type": type, "value": value, "style": FORM_STYLES["input"]}
    if min is not None:
        kwargs["min"] = min
    if max is not None:
        kwargs["max"] = max
    if step is not None:
        kwargs["step"] = step
    if placeholder is not None:
        kwargs["placeholder"] = placeholder
    return html.Div(
        [
            html.Label(label, style=FORM_STYLES["label"]),
            dcc.Input(**kwargs),
        ],
        style={"display": "flex", "flexDirection": "column"},
    )


def dropdown_row(id: str, label: str, options: list[dict], value: str) -> html.Div:
    """Single labeled dropdown."""
    return html.Div(
        [
            html.Label(label, style=FORM_STYLES["label"]),
            dcc.Dropdown(id=id, options=options, value=value, style=FORM_STYLES["input"]),
        ],
        style={"display": "flex", "flexDirection": "column"},
    )


def form_section(children: list, *, input_row_style: dict | None = None) -> html.Div:
    """Wrap form inputs in a section with optional grid style."""
    style = dict(FORM_STYLES["formSection"])
    if input_row_style:
        style.update(input_row_style)
    return html.Div(children, style=FORM_STYLES["formSection"])


def primary_button(id: str, text: str = "Submit") -> html.Button:
    """Primary action button."""
    return html.Button(text, id=id, n_clicks=0, style=FORM_STYLES["button"])


def result_card(children: list, id: str = "output") -> html.Div:
    """Card for displaying result text (e.g. Pre or error)."""
    return html.Div(id=id, children=children, style=RESULT_STYLES["card"])


def result_pre(text: str) -> html.Pre:
    """Preformatted result text."""
    return html.Pre(text, style=RESULT_STYLES["pre"])


def result_error(message: str) -> html.Pre:
    """Error message block."""
    return html.Pre(message, style=RESULT_STYLES["error"])


__all__ = [
    "dropdown_row",
    "form_section",
    "input_row",
    "primary_button",
    "result_card",
    "result_error",
    "result_pre",
]
