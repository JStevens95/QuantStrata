"""
FX Vanilla Option Pricing Calculator — Dash app.

Run from project root:
    python -m src.ui.pricing_calculator

Or in code:
    from src.ui.pricing_calculator import create_app
    app = create_app()
    app.run_server(debug=True)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when run as __main__
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

try:
    import dash
    from dash import dcc, html, Input, Output, State
except ImportError as e:
    raise ImportError("Dash is required for the pricing calculator UI. Install with: pip install dash") from e

from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer


def _price_fx_vanilla(
    spot: float,
    strike: float,
    vol: float,
    r_domestic: float,
    r_foreign: float,
    expiry: float,
    notional: float,
    option_type: str,
) -> tuple[float, dict]:
    """Build market and option, price with BSM, return (pv, greeks_dict)."""
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
    dom_id = MarketId("IR", "CURVE", "USD.OIS")
    for_id = MarketId("IR", "CURVE", "EUR.OIS")

    market = Market(
        asof="2026-01-27",
        quotes={spot_id: Quote(value=spot)},
        curves={
            dom_id: FlatZeroRateCurve(continuously_compounded_rate=r_domestic),
            for_id: FlatZeroRateCurve(continuously_compounded_rate=r_foreign),
        },
        vols={vol_id: FlatVolSurface(sigma=vol)},
    )
    option = FxVanillaEuropeanOption(
        option_type=option_type,
        notional=notional,
        strike=strike,
        expiry=expiry,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=dom_id,
        foreign_curve_id=for_id,
    )
    pricer = FxVanillaEuropeanOptionBsmPricer()
    pv = pricer.price(option, market)
    greeks = pricer.greeks(option, market)
    return pv, dict(greeks)


def create_app() -> dash.Dash:
    """Create and return the Dash app for the FX vanilla pricing calculator."""
    app = dash.Dash(__name__, title="QuantStrata — FX Vanilla Pricing")
    app.layout = html.Div(
        [
            html.H1("FX Vanilla Option Pricing (BSM)"),
            html.Div(
                [
                    html.Label("Spot"),
                    dcc.Input(id="spot", type="number", value=1.08, min=0.01, step=0.01),
                    html.Label("Strike"),
                    dcc.Input(id="strike", type="number", value=1.10, min=0.01, step=0.01),
                    html.Label("Vol (e.g. 0.10)"),
                    dcc.Input(id="vol", type="number", value=0.10, min=0.001, step=0.01),
                    html.Label("Domestic rate"),
                    dcc.Input(id="r_dom", type="number", value=0.05, step=0.01),
                    html.Label("Foreign rate"),
                    dcc.Input(id="r_for", type="number", value=0.03, step=0.01),
                    html.Label("Expiry (years)"),
                    dcc.Input(id="expiry", type="number", value=0.5, min=0.0, step=0.1),
                    html.Label("Notional"),
                    dcc.Input(id="notional", type="number", value=1_000_000, min=1, step=10000),
                    html.Label("Option type"),
                    dcc.Dropdown(id="option_type", options=[{"label": "Call", "value": "call"}, {"label": "Put", "value": "put"}], value="call"),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px", "maxWidth": "400px", "marginBottom": "20px"},
            ),
            html.Button("Price", id="price-button", n_clicks=0),
            html.Div(id="output", style={"marginTop": "20px", "fontFamily": "monospace"}),
        ],
        style={"padding": "20px"},
    )

    @app.callback(
        Output("output", "children"),
        Input("price-button", "n_clicks"),
        State("spot", "value"),
        State("strike", "value"),
        State("vol", "value"),
        State("r_dom", "value"),
        State("r_for", "value"),
        State("expiry", "value"),
        State("notional", "value"),
        State("option_type", "value"),
    )
    def on_price(_n_clicks, spot, strike, vol, r_dom, r_for, expiry, notional, option_type):
        if None in (spot, strike, vol, r_dom, r_for, expiry, notional, option_type):
            return html.Pre("Fill all inputs.")
        try:
            pv, greeks = _price_fx_vanilla(
                spot=float(spot),
                strike=float(strike),
                vol=float(vol),
                r_domestic=float(r_dom),
                r_foreign=float(r_for),
                expiry=float(expiry),
                notional=float(notional),
                option_type=str(option_type),
            )
            lines = [f"PV:    {pv:,.2f}", f"Delta: {greeks.get('delta', 0):.4f}", f"Gamma: {greeks.get('gamma', 0):.6f}", f"Vega:  {greeks.get('vega', 0):.2f}"]
            return html.Pre("\n".join(lines))
        except Exception as e:
            return html.Pre(f"Error: {e!s}")

    return app


def main() -> None:
    app = create_app()
    app.run_server(debug=True, port=8050)


if __name__ == "__main__":
    main()
