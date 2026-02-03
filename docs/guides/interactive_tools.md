# Interactive Tools (Dash UIs)

QuantStrata provides optional **Plotly Dash** UIs under `src/ui/` for browser-based interaction (e.g. pricing calculator). These require the optional UI dependencies.

---

## Setup

Install the UI dependencies from the project root:

```bash
pip install -r requirements-ui.txt
```

This installs `dash` (Plotly Dash). The rest of the library (e.g. pricers, market data) uses the main `requirements.txt`.

---

## Pricing calculator

The **FX Vanilla Pricing Calculator** lets you enter spot, strike, vol, rates, expiry, notional, and option type (call/put) and see the BSM price and Greeks.

### Run from the command line

From the **repository root** (so that `src` is on the path):

```bash
python -m src.ui.pricing_calculator
```

Then open http://127.0.0.1:8050 in your browser.

### Run from code

```python
from src.ui.pricing_calculator import create_app

app = create_app()
app.run_server(debug=True, port=8050)
```

---

## Adding more Dash apps

New UIs can be added under `src/ui/` as separate modules (e.g. `src/ui/risk_dashboard.py`). Each app should expose a `create_app()` that returns a `dash.Dash` instance. Document new apps in this guide and, if useful, add a `__main__` block so they can be run with `python -m src.ui.<module>`.
