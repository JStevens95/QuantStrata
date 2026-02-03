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

## Layout

UI apps live under `src/ui/apps/`, with shared layout, styles, and components in `src/ui/_shared/`. Run any registered app with:

```bash
python -m src.ui.run <app_name>
```

See `src/ui/README.md` for the full folder structure and how to add new apps.

---

## Pricing calculator

The **FX Vanilla Pricing Calculator** (`pricing_calculator`) lets you enter spot, strike, vol, rates, expiry, notional, and option type (call/put) and see the BSM price and Greeks.

### Run from the command line

From the **repository root**:

```bash
python -m src.ui.run pricing_calculator
```

Then open http://127.0.0.1:8050 in your browser. Optional: `python -m src.ui.run pricing_calculator --port 8051 --debug`.

### Run from code

```python
from src.ui.apps.pricing_calculator import create_app

app = create_app()
app.run_server(debug=True, port=8050)
```

Or via the package helper:

```python
from src.ui import create_pricing_calculator_app

app = create_pricing_calculator_app()
app.run_server(debug=True, port=8050)
```

---

## Adding more Dash apps

1. Create `src/ui/apps/<name>/` with `__init__.py` and `app.py` defining `create_app() -> dash.Dash`.
2. Use `_shared` (layout, styles, components) for a consistent look.
3. Register in `src/ui/run.py`: add the app to `APP_REGISTRY`.
4. Document the app in this guide.
