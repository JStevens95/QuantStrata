# Building Dash UIs: Tutorial

This tutorial walks through **running** QuantStrata Dash apps, **building** a new app using the shared layout and components, and **tuning** styling and layouts. For a quick reference on setup and available apps, see [Interactive Tools](interactive_tools.md). For the folder layout and run command, see `src/ui/README.md`.

---

## 1. Running an existing app

Install the UI dependencies from the repository root:

```bash
pip install -r requirements-ui.txt
```

Start the FX Vanilla Pricing Calculator:

```bash
python -m src.ui.run pricing_calculator
```

Open **http://127.0.0.1:8050** in your browser. Optional flags:

- `--port 8051` — use a different port
- `--debug` — enable Dash debug mode (e.g. hot reload)

To understand how the app is built, open `src/ui/apps/pricing_calculator/app.py`. It uses the shared layout, styles, and components described below.

---

## 2. How the UI is structured

All Dash apps live under `src/ui/` and share the same entry point and building blocks.

### Entry point: `run.py`

- **Single command** for any app: `python -m src.ui.run <app_name>`
- `run.py` maintains an `APP_REGISTRY` mapping app names to `(module_path, create_app)`.
- Adding a new app means: (1) implementing `create_app()` in `apps/<name>/app.py`, (2) registering it in `APP_REGISTRY`.

### Shared building blocks: `_shared/`

Every app can reuse:

| Module        | Purpose |
|---------------|---------|
| **layout.py** | `make_app_layout(main_content, app_title=...)` — navbar (brand + app title), main content area, footer. |
| **styles.py** | Style dicts: `LAYOUT_STYLES`, `NAVBAR_STYLES`, `FORM_STYLES`, `RESULT_STYLES`, `FOOTER_STYLES`. Use these in `style=` for a consistent look. |
| **components.py** | Reusable UI pieces: `input_row`, `dropdown_row`, `form_section`, `primary_button`, `result_card`, `result_pre`, `result_error`. |

Using these keeps all apps visually consistent and speeds up development.

### Per-app code: `apps/<name>/`

- **`app.py`** — defines `create_app() -> dash.Dash`: builds the layout (using `_shared`) and registers callbacks.
- **`__init__.py`** — re-exports `create_app` so you can run or import the app by name.

---

## 3. Building a new app (step by step)

### Step 1: Create the app package

Create the folder and files:

```
src/ui/apps/my_app/
├── __init__.py
└── app.py
```

In **`__init__.py`**:

```python
from src.ui.apps.my_app.app import create_app

__all__ = ["create_app"]
```

### Step 2: Implement `create_app()` in `app.py`

Use the shared layout and components so your app looks like the rest of the library.

**Create the Dash instance and layout:**

```python
import dash
from dash import html, Input, Output, State

from src.ui._shared.layout import make_app_layout
from src.ui._shared.styles import FORM_STYLES
from src.ui._shared.components import (
    form_section,
    input_row,
    primary_button,
    result_card,
    result_pre,
    result_error,
)

def create_app():
    app = dash.Dash(__name__, title="QuantStrata — My App")

    form = form_section(
        [
            html.Div(
                [
                    input_row("x", "Input X", value=1.0, min=0, step=0.1),
                    input_row("y", "Input Y", value=2.0, min=0, step=0.1),
                ],
                style=FORM_STYLES["inputRow"],
            ),
            primary_button("compute-button", "Compute"),
            result_card([], id="output"),
        ]
    )

    app.layout = make_app_layout([form], app_title="My App")

    # Callback here (see Step 3)
    return app
```

- **`make_app_layout([form], app_title="My App")`** — wraps your content in the standard navbar, main area, and footer.
- **`form_section([...])`** — groups the form block; pass a list of components (grid of inputs, button, output div).
- **`html.Div(..., style=FORM_STYLES["inputRow"])`** — lays out the inputs in a responsive grid (see [Tuning layouts](#42-form-grid) below).
- **`input_row(id, label, value=..., min=..., step=...)`** — single labeled input; use **`dropdown_row(id, label, options, value)`** for dropdowns.
- **`primary_button(id, text)`** — primary action button.
- **`result_card([], id="output")`** — container for callback output; your callback will return `result_pre(...)` or `result_error(...)`.

### Step 3: Add a callback

In the same `app.py`, register a callback that reads from your inputs and updates the result card:

```python
@app.callback(
    Output("output", "children"),
    Input("compute-button", "n_clicks"),
    State("x", "value"),
    State("y", "value"),
)
def on_compute(_n_clicks, x, y):
    if x is None or y is None:
        return result_pre("Fill all inputs.")
    try:
        # Your logic here
        result = float(x) + float(y)
        return result_pre(f"Result: {result}")
    except Exception as e:
        return result_error(f"Error: {e!s}")
```

- **Output** — `"output"` is the `id` of `result_card(..., id="output")`; you replace its `children`.
- **Input** — the button’s `n_clicks` triggers the callback.
- **State** — values from inputs without triggering on every keystroke.
- Return **`result_pre(text)`** for normal output and **`result_error(message)`** for errors so styling stays consistent.

### Step 4: Register the app in `run.py`

Open `src/ui/run.py` and add your app to `APP_REGISTRY`:

```python
APP_REGISTRY = {
    "pricing_calculator": ("src.ui.apps.pricing_calculator.app", "create_app"),
    "my_app": ("src.ui.apps.my_app.app", "create_app"),
}
```

Then run:

```bash
python -m src.ui.run my_app
```

Document the new app in [Interactive Tools](interactive_tools.md) and, if useful, in `src/ui/README.md`.

---

## 4. Tuning styling and layouts

All shared style dicts live in **`src/ui/_shared/styles.py`**. Changing them affects every app that uses them. For app-specific tweaks, merge or override these dicts in your `app.py`.

### 4.1 Global layout and navbar

- **`LAYOUT_STYLES`**
  - **`container`** — outer wrapper: `maxWidth`, `padding`, `fontFamily`, `margin`.
  - **`main`** — main content block: e.g. `marginTop`.
- **`NAVBAR_STYLES`**
  - **`navbar`** — bar: `borderBottom`, `padding`, `marginBottom`.
  - **`brand`** — brand text: `fontSize`, `fontWeight`, `color`.
  - **`appTitle`** — app subtitle: `fontSize`, `color`, `marginLeft`.

To change the shell (width, font, navbar look), edit these in `styles.py`. The layout is applied in **`_shared/layout.py`** via `make_app_layout()`; you can pass a custom **`brand`** if needed:

```python
make_app_layout([form], app_title="My App", brand="My Quant Library")
```

### 4.2 Form grid

Form fields are laid out in a grid using **`FORM_STYLES["inputRow"]`**:

- **`display": "grid"`**
- **`gridTemplateColumns": "repeat(auto-fill, minmax(140px, 1fr))"`** — responsive columns (~140px min).
- **`gap": "12px"`**, **`alignItems": "end"`**, **`maxWidth": "800px"`**

To get a fixed number of columns (e.g. 4), override when building the layout in your app:

```python
from src.ui._shared.styles import FORM_STYLES

grid_style = {**FORM_STYLES["inputRow"], "gridTemplateColumns": "repeat(4, 1fr)"}
html.Div([input_row(...), ...], style=grid_style)
```

To make the form wider, increase **`maxWidth`** in `inputRow` (in `styles.py`) or in a local override as above.

### 4.3 Inputs and buttons

- **`FORM_STYLES["label"]`** — label text (size, weight, color).
- **`FORM_STYLES["input"]`** — input/dropdown (padding, border, radius, font).
- **`FORM_STYLES["button"]`** — primary button (background, padding, radius, cursor).

These are used inside **`_shared/components.py`** (`input_row`, `dropdown_row`, `primary_button`). To change how all inputs or buttons look, edit `styles.py`. For a one-off change in one app, you can pass a custom `style` into components if they support it, or wrap elements in an `html.Div` with your own style.

### 4.4 Result block

- **`RESULT_STYLES["card"]`** — container for results: border, radius, padding, background.
- **`RESULT_STYLES["pre"]`** — normal result text (monospace, wrap).
- **`RESULT_STYLES["error"]`** — error text (e.g. red, monospace).

Used by **`result_card`**, **`result_pre`**, and **`result_error`** in `components.py`. Adjust in `styles.py` for global changes.

### 4.5 Footer

- **`FOOTER_STYLES["footer"]`** — margin, padding, border, font size, color.

Defined in `styles.py`, applied in `layout.py`. Edit there to change the footer for all apps.

---

## 5. Summary

| Goal | Action |
|------|--------|
| **Run an app** | `pip install -r requirements-ui.txt` then `python -m src.ui.run <app_name>` |
| **Build a new app** | Add `apps/<name>/` with `create_app()` using `make_app_layout`, `form_section`, `input_row` / `dropdown_row`, `primary_button`, `result_card` / `result_pre` / `result_error`, and register in `run.py`. |
| **Tune layout** | Edit `_shared/styles.py` (`LAYOUT_STYLES`, `NAVBAR_STYLES`, `FORM_STYLES`, etc.) or override in app with `style={**FORM_STYLES["key"], ...}`. |
| **Tune form grid** | Override `FORM_STYLES["inputRow"]` (e.g. `gridTemplateColumns`, `maxWidth`) when building the form in your app. |

For more detail on the folder structure and run options, see **`src/ui/README.md`**. For setup and the list of apps, see [Interactive Tools](interactive_tools.md).
