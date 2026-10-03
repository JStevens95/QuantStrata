# Rade Analytics — Page Template

**Status:** v1 · **Scope:** new pages under `src/ui/apps/rade_analytics/` ·
**Pairs with:** [`page_contract.md`](../page_contract.md)

## What this is

A pair of copy-paste-ready Python files that capture the canonical shape every
page in the Rade Analytics UI follows.  Use this when adding a new page —
**Cross-Cluster, Data Quality, Trade Graph rebuild, AI Assistant, etc.**

The two template files mirror the existing `Cluster Deep-Dive` reference
implementation, which is the most contract-compliant page in the app:

| Template file | Maps to | Production reference |
|---|---|---|
| [`template_layout.py`](./template_layout.py) | `layouts/<your_page>.py` | `layouts/evaluation/cluster_deep_dive.py` |
| [`template_cb.py`](./template_cb.py) | `callbacks/<your_page>_cb.py` | `callbacks/cluster_deep_dive_cb.py` |

> **These files do not run.**  They are documentation-shaped Python so your
> editor gives you syntax highlighting / jump-to-definition.  They live under
> `docs/` so the app never imports them.

---

## 15-minute add-a-page workflow

### 1.  Copy + rename (1 min)

```bash
cp docs/rade_analytics/page_template/template_layout.py \
   src/ui/apps/rade_analytics/layouts/<your_page>.py
cp docs/rade_analytics/page_template/template_cb.py \
   src/ui/apps/rade_analytics/callbacks/<your_page>_cb.py
```

### 2.  Search-replace the markers (2 min)

| Marker | Replace with |
|---|---|
| `_TEMPLATE_PATH` | `/your-route` (e.g. `/evaluation/cross-cluster`) |
| `TEMPLATE_IDS` | `<YOUR_PAGE>_IDS` (e.g. `CROSS_CLUSTER_IDS`) |
| `_template_*` private symbols | renamed to `_<your_page>_*` |
| `build_template` | `build_<your_page>` |
| `# TODO:` markers | implementation per the comment above each |

### 3.  Wire into the router + sidebar (3 min)

* Add an entry to the route table in `router.py` (or to the sub-tab spec
  list in `layouts/evaluation/shell.py` for an Evaluation child).
* Register the page in `app.py`:

  ```python
  from .callbacks import your_page_cb
  your_page_cb.register(app, backend)
  ```

* Add a navigation link in `components/sidebar.py`.

### 4.  Implement data hooks (5 min — the actual page work)

* Add a backend method to `data/backend.py` that returns
  `BackendResult[<your DTO>]`.  Follow the cache-key convention used by
  every other `RadeBackend.*` method.
* Replace the `# TODO: backend lookup` line in the bootstrap callback.
* Replace the `# TODO: render <metric>` lines in each render callback.

### 5.  Smoke + ship (4 min)

```bash
pytest tests/ui/apps/rade_analytics/ -k <your_page> -x
.venv/bin/python -c "from src.ui.apps.rade_analytics.app import create_app; create_app()"
```

---

## Helper cheat sheet

These two helpers collapse most of the tri-state boilerplate every render
callback used to hand-roll.  They live in
[`src/ui/apps/rade_analytics/data/result_helpers.py`](../../../src/ui/apps/rade_analytics/data/result_helpers.py)
and are re-exported from `..data` for short imports.

### `figure_with_fallback`

```python
from ..data import figure_with_fallback
from ..figures import portfolio_pnl

return figure_with_fallback(
    backend.portfolio_timeseries_df(split, filters),
    on_ok=lambda df: portfolio_pnl(df, uirevision_key=split),
    empty_msg="No portfolio data for the active filter set.",
)
```

Tri-state branches:

| State | Result |
|---|---|
| `not res.ok` | `empty_figure(error_msg or f"Error: {res.error}")` |
| `res.ok` but data is empty | `empty_figure(empty_msg)` |
| `res.ok` and data is populated | `on_ok(res.data)` |

### `component_with_fallback`

```python
from ..data import component_with_fallback
from ..components.kpi_card import KpiCard

return component_with_fallback(
    backend.cluster_kpis(cluster_id=cid),
    on_ok=lambda kpis: KpiCard(label="MAE", value=f"{kpis.mae:.4f}"),
    empty_title="No KPIs staged for this cluster",
    error_title="Couldn't load cluster KPIs",
    on_retry_id="kpi-retry",
)
```

Tri-state branches: same shape as the figure helper, but routes through
`Empty` / `Error` from `components.state_wrappers` for layout-level outputs.

### When the helper doesn't fit

Render callbacks emitting **multiple Outputs of different types from a
single BackendResult** still need an explicit branch — the helper covers
single-output → single-state cases.  Pattern:

```python
res = backend.cluster_kpis(cluster_id=cid)
if not res.ok or res.data is None:
    placeholder = "—"
    empty_fig = empty_figure("…")
    return placeholder, placeholder, empty_fig

mae_txt = f"{res.data.mae:.4f}"
rmse_txt = f"{res.data.rmse:.4f}"
fig = some_chart(res.data, uirevision_key=cluster_id)
return mae_txt, rmse_txt, fig
```

---

## Pre-flight checklist (mirrors page_contract.md §3)

Before raising the PR, verify:

- [ ] Layout is **pure** — `build_<page>(*, session)` returns the layout
      with all initial values seeded from `session`; no hydration callback
      after mount.
- [ ] `mount_signal` Store is in the layout root with `data=True` (Rule L4)
      and is the **trigger `Input`** for the bootstrap callback **and every
      page-scoped render callback** (Rule C7).
- [ ] No render callback uses `Input(SHELL_IDS["url"], "pathname")` —
      pathname is `State`-only on every page-scoped callback (Rule C4 +
      Anti-pattern A8). `grep -nE 'Input\(\s*SHELL_IDS\["url"\]\s*,\s*"pathname"\)'
      src/ui/apps/rade_analytics/callbacks/<your_page>_cb.py` returns no
      hits.
- [ ] Module docstring lists capture and render callbacks under separate
      banners (`§N. <name>`).
- [ ] `register()` delegates to `_register_capture(app)` and
      `_register_render(app, backend)` only.  No callbacks defined directly
      in `register()`.
- [ ] Every render callback gates on `pathname != _<PAGE>_PATH`.
- [ ] Every render callback that consumes a `BackendResult` either uses
      `figure_with_fallback` / `component_with_fallback`, or branches
      tri-state explicitly.
- [ ] Every time-series figure passes `uirevision_key=...` keyed on the
      data domain (e.g. `split`, `cluster_id`, `f"{split}::{cluster_id}"`).
- [ ] No `Output` is registered twice without an `allow_duplicate=True`
      comment justifying it.
- [ ] No callback uses `Input(other_callback_output)` — collapse it or
      move the writer logic into the layout.
- [ ] `prevent_initial_call` matches the role-based table in Rule C5:
      `True` for capture; `"initial_duplicate"` for `mount_signal`-driven
      bootstrap + render. **`False` is forbidden on page-scoped callbacks.**
- [ ] Smoke: `from src.ui.apps.rade_analytics.app import create_app;
      create_app()` boots without warnings.
- [ ] Lint clean — no unused imports, no orphan helpers.

---

## Updating the template

If a future page introduces a pattern worth canonising — e.g. a different
shape for "page with no Evaluation filter bar" — update the template files
*and* `page_contract.md` together.  The two are meant to drift in lock-step;
the audit checklist in `page_contract.md` is the source of truth, the
template is the executable form.
