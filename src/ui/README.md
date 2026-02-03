# QuantStrata UI — Dash apps

Optional **Plotly Dash** UIs for browser-based interaction (e.g. pricing calculator). Install dependencies with:

```bash
pip install -r requirements-ui.txt
```

## Layout

```
src/ui/
├── __init__.py          # create_pricing_calculator_app() and package API
├── README.md            # This file
├── run.py               # Entry point: python -m src.ui.run <app_name>
├── _shared/             # Shared building blocks for all apps
│   ├── layout.py        # Common layout (navbar, footer, container)
│   ├── styles.py        # CSS / style constants
│   └── components.py    # Reusable components (inputs, cards, buttons)
└── apps/
    └── pricing_calculator/
        ├── __init__.py
        └── app.py       # create_app(), layout, callbacks
```

## Run an app

From the **repository root**:

```bash
python -m src.ui.run pricing_calculator
```

Then open http://127.0.0.1:8050. Optional: `--port 8051`, `--debug`.

## Use in code

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

## Adding a new app

1. Create `src/ui/apps/<name>/` with `__init__.py` and `app.py`.
2. In `app.py`, define `create_app() -> dash.Dash` (use `_shared` layout and components).
3. Register in `run.py`: add `"<name>": ("src.ui.apps.<name>.app", "create_app")` to `APP_REGISTRY`.
4. Document the app in `docs/guides/interactive_tools.md`.
