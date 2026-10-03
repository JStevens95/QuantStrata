# Rade Analytics — Page Contract

**Status:** draft v1 · **Scope:** `src/ui/apps/rade_analytics/**` · **Owner:** UI lead

This is the single source of truth for **how pages are designed, built and
wired** in the Rade Analytics UI. Every page — Overview, Evaluation
sub-tabs, Monitoring, Governance, Inference, Scenario Lab, Report Builder,
Data Quality, AI Assistant — is built against this contract.

The motivation: as of v1 the codebase has the right *layered* architecture
but inconsistent *intra-layer* discipline. Three days of "phantom" callback
errors, multi-writer races and stale-DOM warnings on the Evaluation page
all trace back to the same handful of unstated conventions being violated.
This document states them, names the patterns from production-grade Dash
practice, and gives a pre-flight checklist for any PR.

> **Adding a new page?** Don't read this end-to-end first.  Start at
> [`page_template/README.md`](./page_template/README.md) — it gives you a
> 15-minute copy-paste workflow that already implements every contract
> rule below.  Come back to this document when you hit a question the
> template doesn't answer or when you're auditing an existing page.

---

## 0. TL;DR

If you only read one section, read this:

1. **Layout is `pure(session)`.** Initial values for inputs are baked in at
   build time, not pushed by a hydration callback after mount.
2. **Mount tripwire is mandatory.** Every page layout root carries a
   `dcc.Store(id=mount_signal, data=True)`. Bootstrap and every
   page-scoped render callback trigger off it (Rule L4 + Rule C7).
3. **One writer per `Output` by default.** `allow_duplicate=True` requires a
   one-line code comment justifying the second writer.
4. **No input→input chains across callbacks.** If callback A writes a prop
   B reads as `Input`, collapse them or move A's logic into the layout.
5. **Capture and render are separate callbacks.** Capture writes session;
   render writes UI. One callback never does both.
6. **Page-scoped callbacks gate on `pathname` as `State`.** Pathname
   *gates*, never *triggers*. `Input(pathname)` on a page-scoped callback
   creates a mount race against the router's content swap — Anti-pattern A8.
7. **`prevent_initial_call`** is dictated by callback role (Rule C5):
   `True` for capture, `"initial_duplicate"` for mount-tripwire-driven
   bootstrap + render. `False` is **forbidden** on page-scoped callbacks.
8. **Build pages in five PR-sized steps.** Static-from-mock → capture →
   render → live-data → perf. Don't merge them.

The rest of this document expands each rule with rationale, anti-patterns
we've already hit and concrete code conventions taken from existing
modules.

---

## 1. Architectural layers

The codebase splits into six layers. Each layer has a single job and is
allowed to depend only on the layers below it.

```
┌────────────────────────────────────────────────────────────────────┐
│  app.py / router.py        — wire-up, top-level URL routing        │
├────────────────────────────────────────────────────────────────────┤
│  callbacks/<page>_cb.py    — Dash callbacks (capture + render)     │
├────────────────────────────────────────────────────────────────────┤
│  layouts/<page>.py         — pure layout builders                  │
├────────────────────────────────────────────────────────────────────┤
│  components/<thing>.py     — reusable primitives + ID dicts        │
│  figures/<chart>.py        — plotly figure factories               │
├────────────────────────────────────────────────────────────────────┤
│  data/session.py           — typed user-session state              │
│  data/backend.py           — RadeBackend + BackendResult           │
├────────────────────────────────────────────────────────────────────┤
│  assets/                   — auto-loaded static + JS               │
│    rade.css                  styles                                │
│    js/<page>.js              clientside_callback functions         │
├────────────────────────────────────────────────────────────────────┤
│  src/rade_ml_pt/ensemble/  — FastAPI server + raw API client       │
│  api/                                                              │
└────────────────────────────────────────────────────────────────────┘
```

**`assets/` convention**: Dash auto-loads every file in this directory at
startup. CSS goes in `rade.css` (single file, scoped via class prefixes).
Clientside callback functions live in `assets/js/<page>.js` — one file
per page namespace, never inline JS strings in Python (Rule P2 below).

Allowed dependencies are downward-only:

- `layouts/` may import `components/`, `figures/`, `data/session.py`.
  Layouts **must not** import `callbacks/` or `data/backend.py`.
- `callbacks/` may import everything below. Callbacks **must not** import
  another `callbacks/<page>_cb.py` module.
- `components/` may import `data/session.py` (for type hints) but **must
  not** import `data/backend.py` — components are framework primitives,
  not page-aware.
- `figures/` may import nothing from `layouts/` or `callbacks/`. They take
  data in, return `go.Figure` out.

### Why this matters

The hard line between `layouts/` (no backend) and `callbacks/` (yes
backend) is what lets us follow the build workflow in §7. A layout PR
that touches `data/backend.py` is a smell — it means the layout is doing
data fetching that should have been a callback.

---

## 2. Page module contract

Every page consists of **exactly two modules**: a layout module under
`layouts/<page>.py` and a callback module under `callbacks/<page>_cb.py`.
For pages with sub-tabs (Evaluation), the layout module is a package
(`layouts/<page>/`) with one file per sub-tab plus a `shell.py`.

### 2.1 Layout module

```
src/ui/apps/rade_analytics/layouts/<page>.py
```

Every layout module **must** export:

- An `<PAGE>_IDS: Dict[str, str]` mapping intent name → DOM id. All ids
  used by callbacks live here. Hardcoded id strings outside this dict
  are forbidden.
- A `build_<page>(session: Session, ...) -> html.Div` function that
  returns the entire page tree. Pure function of its arguments.
- An `__all__ = [...]` list at module foot.

```python
# layouts/<page>.py
from __future__ import annotations

from typing import Any
from dash import html
from ..data.session import Session

<PAGE>_IDS = {
    "root":        "<page>-root",
    # ... all dynamic ids ...
}

def build_<page>(session: Session) -> html.Div:
    """Build the <page> page tree from session state.

    Pure function — no callbacks, no backend access, no module-level
    state.  Initial values for every input come from ``session``.
    """
    return html.Div(
        id=<PAGE>_IDS["root"],
        className="rade-page rade-<page>",
        children=[...],
    )

__all__ = ["<PAGE>_IDS", "build_<page>"]
```

### 2.2 Callback module

```
src/ui/apps/rade_analytics/callbacks/<page>_cb.py
```

Every callback module **must** export a single `register(app, backend)`
function. Inside, callbacks are grouped into clearly-labelled sections —
**capture** (user → session) first, **render** (session → UI) second:

```python
# callbacks/<page>_cb.py
from __future__ import annotations

from typing import TYPE_CHECKING
from dash import Input, Output, State

from ..layouts.<page> import <PAGE>_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash
    from ..data.backend import RadeBackend


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every <page> callback to ``app``."""
    _register_capture(app)
    _register_render(app, backend)


# ─────────────────────────────────────────────────────────────────
# Capture — user actions → session.data
# ─────────────────────────────────────────────────────────────────

def _register_capture(app: "Dash") -> None:
    ...


# ─────────────────────────────────────────────────────────────────
# Render — session.data → UI props
# ─────────────────────────────────────────────────────────────────

def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    ...


__all__ = ["register"]
```

This split is **enforced**. A callback that does both capture and render
in one body is a code-review reject.

---

## 3. Layout contract

### Rule L1 — `build_<page>(session)` is a pure function

The layout function reads everything it needs from `session` (and
optionally from a `backend` for *static* options like dropdown choices
that are version-keyed and cached). It returns a fully-initialised tree
with every input pre-populated. **No post-mount hydration callbacks.**

The Evaluation filter bar is the canonical anti-pattern that motivated
this rule:

```python
# DON'T — registered a separate _hydrate_filter_bar callback that wrote
# session values into MultiSelects after mount.  Created a race between
# chrome rebuild and value writes; created a chained _sync_filters_to_session
# call attempting to write to a not-yet-mounted toggle label.

@app.callback(
    Output(EVAL_FILTER_IDS["asset_class"], "value", allow_duplicate=True),
    ...
    Input(SHELL_IDS["url"], "pathname"),
)
def _hydrate_filter_bar(...): ...
```

```python
# DO — initial values baked into the layout build, no callback needed.

def build_evaluation_filter_bar(
    *,
    initial_filters: EvaluationFilters,  # ← from session at build time
    ...
) -> html.Div:
    return html.Div(
        ...
        children=[
            dmc.MultiSelect(
                id=EVAL_FILTER_IDS["asset_class"],
                value=list(initial_filters.asset_class),  # ← here
                ...
            ),
            ...
        ],
    )
```

The router (or whatever calls `build_<page>`) reads `session` once and
passes it through. After mount, the only thing that writes to a UI input
prop is the **render** callback chain in §4.

### Rule L2 — IDs live in `<PAGE>_IDS` dicts

Hardcoded string ids in either layouts or callbacks are forbidden. The
`<PAGE>_IDS` dict is the contract between the layout (which uses them as
component ids) and the callbacks (which use them as `Input` / `Output` /
`State` targets). Renaming an id is a one-line edit; finding all
references is a one-grep operation.

```python
# layouts/<page>.py
<PAGE>_IDS = {
    "root":         "<page>-root",
    "filter_pane":  "<page>-filter-pane",
    "main_chart":   "<page>-main-chart",
    "kpi_strip":    "<page>-kpi-strip",
    "trades_grid":  "<page>-trades-grid",
}
```

### Rule L3 — Visual styling lives in `rade.css`, not inline

Callbacks read styling by intent. When a button needs to show / hide,
flip its className via a constant defined at the top of the callback
module:

```python
# callbacks/splash_cb.py — existing pattern
_DOT_OK    = "rade-status-dot rade-status-dot--ok"
_DOT_ERR   = "rade-status-dot rade-status-dot--err"
_DOT_BOOT  = "rade-status-dot rade-status-dot--booting"
```

Inline `style={"color": "red"}` is a code-review reject. The exception
is `style={"display": "none"}` for visibility toggles where the absent /
present binary doesn't warrant a class.

### Rule L4 — Mount tripwire is mandatory on every page

Every page layout root contains exactly one mount-tripwire Store:

```python
# layouts/<page>.py
return html.Div(
    id=<PAGE>_IDS["root"],
    children=[
        ...
        # Mount tripwire — Page Contract §3 Rule L4.  ``data=True`` at
        # build time fires the bootstrap + every page-scoped render
        # callback exactly once per fresh mount of the page.  Lives
        # *inside* the layout chunk the router swaps in, so it can't
        # race the router's content swap on cross-page nav.
        dcc.Store(
            id=<PAGE>_IDS["mount_signal"],
            data=True,
            storage_type="memory",
        ),
    ],
)
```

Why a Store on the layout root, not `pathname` or `top_level_store`:

| Trigger candidate | Failure mode |
|---|---|
| `Input(pathname)` | Fires concurrently with the router's content swap; outputs target IDs that haven't been mounted yet → Dash silently drops the writes → empty page on first visit. |
| `Input(top_level_store.data)` | Only fires on cross-top-level navigation (splash ↔ overview ↔ evaluation). Misses within-Evaluation sub-tab transitions. |
| `Input(mount_signal.data)` | Lives **inside** the swapped layout chunk. By the time it appears in the DOM, every Output the dependent callbacks target is also mounted. No race. |

This rule is the layout-side half of Rule C7. The two are inseparable —
the Store has to exist (Rule L4) for callbacks to trigger off it
(Rule C7).

---

## 4. Callback contract

### Rule C1 — One writer per `Output` by default

Every Dash `Output` has exactly one callback writing to it. Two writers
require `allow_duplicate=True` *and* a code comment explaining the
orthogonal write paths. Reset / clear-all buttons writing to dropdown
`value` props are the only common legitimate use case.

```python
# DO — second writer is justified inline
@app.callback(
    Output(IDS["filter_value"], "value", allow_duplicate=True),
    Input(IDS["reset_btn"], "n_clicks"),
    prevent_initial_call=True,
)
def _reset_filter(n):  # noqa: D
    # Justification: orthogonal write path — the user's "Reset"
    # button explicitly clobbers whatever the user typed.  The
    # primary writer is the typing → debounced sync callback.
    ...
```

If you find yourself adding `allow_duplicate=True` without a comment, you
are about to create the kind of multi-writer ambiguity that bit us on
`Collapse.opened`. Stop and merge the callbacks instead.

### Rule C2 — Capture vs render

Every callback is either a **capture** or a **render**. Never both.

| | Capture | Render |
|---|---|---|
| `Input` is | a user action (click, value change, URL change) | `session-store.data` (and possibly fetched data) |
| `Output` is | `session-store.data` only | UI props (figures, AG Grid `rowData`, classNames, `value` of inputs not driven by user) |
| Calls backend | No | Yes (when the page needs server data) |
| Test by | feeding mock user input | feeding mock session dict |

**Why:** capture callbacks have small, deterministic outputs (a JSON dict).
Render callbacks have large outputs (figures, grids) but are pure
functions of session + backend response. Separating them means each is
unit-testable in isolation, and the testing surface is the public
interfaces of `Session` and `RadeBackend` — both of which already exist
and are stable.

### Rule C3 — No input→input chains

If callback A writes to a UI prop (`value`, `data`, `opened`, `figure`)
that callback B reads as an `Input`, you have created a chain. Chains are
the source of every race condition we've fought:

- Mount races: A writes before B's target is in the DOM.
- Lost-update races: B fires before A's write is committed.
- Double-execution: A and B both fire on the same external trigger.

The fix is one of:

1. **Collapse**: merge A and B into a single callback that takes both
   triggers and uses `ctx.triggered_id` to branch.
2. **Move to layout**: if A's role was "push session into UI on mount",
   make the layout do that at build time (Rule L1).
3. **Use `dcc.Store` as a stable middleman**: A writes a derived store,
   B reads the store. Stores are never unmounted (they live in the root
   layout), so there is no mount race.

### Rule C4 — Page-scoped callbacks gate on `pathname` (as `State`)

Every callback whose Outputs target ids inside a route-bound subtree
must start with:

```python
def _my_callback(..., pathname):
    if pathname != _<PAGE>_PATH:
        raise PreventUpdate
    ...
```

This is cheap insurance against the "nonexistent object was used in an
Output" warning. It matters because the router's chrome-rebuild
semantics mean the page-scoped DOM goes through unmount → mount cycles
on cross-top-level navigation, and Dash's input-change events can
straddle that transition.

**`pathname` is always added as a `State`, never as an `Input`** —
this is non-negotiable. The URL must *gate* the callback, never
*trigger* it. Triggering on `Input(pathname)` creates a mount race
(see Rule C7 + Anti-pattern A8): the callback fires the moment
`pathname` changes, which is *before* the router has finished
swapping in the new layout chunk, so the outputs target IDs that
aren't in the DOM yet → Dash silently drops every write → the page
paints empty.

### Rule C5 — `prevent_initial_call` discipline

Every callback declares `prevent_initial_call` explicitly. Omitting it
is forbidden. The legal values, by callback role:

| Callback role | Legal value | Why |
|---|---|---|
| Capture (user → session) | `True` | Capture must not fire on mount; only on real user input. |
| Render driven by `mount_signal` (Rule C7) | `"initial_duplicate"` | Mount-signal is *only* useful if it fires on the initial mount of its layout chunk. Use `"initial_duplicate"` regardless of whether the callback has `allow_duplicate=True` Outputs — it's the form Dash recognises for "fire on initial mount even with duplicate-output siblings". |
| Bootstrap (mount_signal → option list + override edges) | `"initial_duplicate"` | Same reasoning, plus the bootstrap typically writes `session-store.data` which is a duplicate output. |
| Clientside callback | `True` (Dash default for `clientside_callback`) | n/a |

`prevent_initial_call=False` is **forbidden** on any callback whose
Outputs live in a route-bound subtree. It tempts you to use
`Input(pathname)` instead of `Input(mount_signal)` and creates the
race in Anti-pattern A8.

### Rule C6 — Pattern-matching for repeated controls

When a control is repeated (one × button per chip, one row per cluster
in a list), use Dash pattern-matching ids:

```python
id={"type": "eval-filter-chip-close", "dimension": "desk"}
```

Then a single callback with `Input({"type": "...", "dimension": ALL}, ...)`
handles every instance. `ctx.triggered_id` tells you which one fired.
Remember the gotcha: pattern-matching callbacks fire once on initial
mount with `n_clicks_list` all-`None`. Bail with:

```python
if not any(n_clicks_list or []):
    raise PreventUpdate
```

### Rule C7 — Page-scoped render callbacks trigger on `mount_signal`, not `pathname`

The trigger Inputs of every page-scoped render callback are exactly:

1. **`Input(<PAGE>_IDS["mount_signal"], "data")`** — fires once per
   fresh mount, with no race against the router's content swap
   (see Rule L4 for why the Store is the right trigger).
2. **`Input(SHELL_IDS["session_store"], "data")`** — fires when any
   capture callback writes session.
3. *(optional)* `Input(<PAGE>_IDS["store_<thing>"], "data")` — for
   any **page-internal** store the render depends on (e.g.
   `store_trade_types` populated by an upstream render).

Pathname is added as a **`State`** for the Rule C4 gate. URL
deep-link parsing (e.g. `?cid=…`) reads `State(SHELL_IDS["url"], "search")`,
again as State.

```python
@app.callback(
    Output(<PAGE>_IDS["main_chart"], "figure"),
    # Trigger Inputs — Rule C7.
    Input(<PAGE>_IDS["mount_signal"],     "data"),
    Input(SHELL_IDS["session_store"],     "data"),
    # Gate State — Rule C4.
    State(SHELL_IDS["url"],               "pathname"),
    # Initial-call discipline — see Rule C5.
    prevent_initial_call="initial_duplicate",
)
def _render(_mount, session_data, pathname):
    if pathname != _<PAGE>_PATH:
        raise PreventUpdate
    ...
```

Why this matters: render callbacks live and die by **whether their
outputs are mounted when the callback fires**. `mount_signal` is the
only trigger that's guaranteed to fire *after* the layout chunk has
been delivered to the browser. Every other candidate (pathname,
top_level_store, raw `prevent_initial_call=False` against an
always-mounted Input) leaves a window where the outputs are missing
and the writes are silently dropped (Anti-pattern A8).

The capture callbacks in §2 are exempt — they target
`session-store.data` (which is always mounted in the shell, never
in a swapped sub-tree) and run `prevent_initial_call=True`, so they
can't race anything.

---

## 5. Data access contract

All page data flows through `RadeBackend`. Callbacks **must not** import
`src.rade_ml_pt.ensemble.api.client` directly — that's a Rule-A1
architectural violation.

### 5.1 Tri-state branching

Every `RadeBackend` method returns `BackendResult[T]`. Render callbacks
branch on `result.ok` and produce one of four UI states:

```python
res = backend.portfolio_df(split=session.split)
if not res.ok:
    return _error_card(res.error)        # ← error
if res.data is None or res.data.empty:
    return _empty_card("No portfolio data for this split")  # ← empty
return _portfolio_figure(res.data)        # ← data
# ← loading is shown via `dcc.Loading` wrapper in the layout
```

The four UI states (loading / data / empty / error) **must all be
visually represented** for any data-driven panel. Showing a blank panel
on error is a UX failure mode and a code-review reject.

### 5.2 Cache awareness

`RadeBackend` is already memoized via `flask_caching`. Callbacks can
call `backend.foo_df(...)` repeatedly without performance worry — only
the first call per `(active_version, args...)` tuple hits the API.

The implication: **don't pre-compute data at layout-build time and stuff
it into a `dcc.Store`**. The cache already does this for you. Layouts
stay synchronous and pure.

### 5.3 DataFrame shape contract

Tabular endpoints return pandas DataFrames; callbacks pass them straight
into `dash_ag_grid` (`rowData=df.to_dict("records")`) or
`plotly.express` / figure factories under `figures/`. Pydantic models
are kept only for compact KPI-style payloads where typed field access is
worth the boilerplate (`HealthResponse`, `OverviewResponse`,
`TradeGraphResponse`). When in doubt, prefer DataFrame.

---

## 6. Performance levers

These are adopted **now**, while pages are still small. The cost of
adding them retrofittedly is multi-day; the cost of starting with them is
zero.

### Lever P1 — `dash.Patch()` for partial figure updates

When a render callback only changes part of a figure (a trace's data,
the layout's title, an annotation), use `dash.Patch()` instead of
returning a fresh `go.Figure`. The patch is sent over the wire as a
diff; the browser applies it locally.

```python
# DO — patch one trace, not the whole figure
@app.callback(Output("portfolio-chart", "figure", allow_duplicate=True),
              Input(...), prevent_initial_call=True)
def _update_predictions_trace(...):
    patch = dash.Patch()
    patch["data"][0]["y"] = new_y_values  # only the predictions trace
    return patch
```

Use the full `go.Figure` rebuild only for the **first** render of a
chart; subsequent updates patch.

### Lever P2 — `clientside_callback` for trivial UI

Anything that's a pure function of inputs already in the browser — chip
counts, "n active" labels, button enabled/disabled states, className
swaps — runs as a `clientside_callback` whose body lives in
**`assets/js/<page>.js`**. No server round-trip, no network latency.

**JS lives in its own file**, never inline in Python. This keeps each
language tooled separately (JS gets editor highlighting / linting), keeps
Python modules readable, and scales as the JS surface grows. Dash
auto-loads everything in `assets/` at startup; reference functions via
`dash.dependencies.ClientsideFunction(namespace=..., function_name=...)`.

```javascript
// assets/js/evaluation.js
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.evaluation = {
    update_filter_label: function(asset_class, currency, desk, product, dateRange) {
        const n = [asset_class, currency, desk, product]
            .filter(arr => Array.isArray(arr) && arr.length > 0).length
            + ((dateRange && (dateRange[0] || dateRange[1])) ? 1 : 0);
        return n > 0 ? `${n} active` : "";
    },
};
```

```python
# callbacks/evaluation_cb.py
from dash import ClientsideFunction

app.clientside_callback(
    ClientsideFunction(namespace="evaluation",
                       function_name="update_filter_label"),
    Output(EVAL_FILTER_IDS["toggle_label"], "children"),
    Input(EVAL_FILTER_IDS["asset_class"], "value"),
    Input(EVAL_FILTER_IDS["currency"], "value"),
    Input(EVAL_FILTER_IDS["desk"], "value"),
    Input(EVAL_FILTER_IDS["product"], "value"),
    Input(EVAL_FILTER_IDS["date_range"], "value"),
)
```

Convention: one JS file per page (`assets/js/evaluation.js`,
`assets/js/overview.js`, …). The namespace matches the filename stem
(`evaluation` ↔ `evaluation.js`). Shared utilities go in
`assets/js/_shared.js` with namespace `shared`.

Rule of thumb: if the callback body fits in five lines of JS and doesn't
touch `backend`, it should be `clientside_callback`.

### Lever P3 — `uirevision` on time-series

Every chart that the user might zoom or pan gets a static `uirevision`
in its `update_layout`. When the figure is re-rendered with new data,
plotly preserves the user's zoom / pan / legend selections.

```python
fig.update_layout(uirevision="portfolio-pnl-stable")
```

The string is per-chart-identity, not per-data-update. Change it only
when you want to *force* a reset (e.g. user switched cluster).

### Levers reserved for later

- **`background=True` callbacks** — adopt when we have a callback that
  takes >2s wall-clock. Today nothing does.
- **`ServersideOutput` / `dash_extensions.enrich`** — adopt when we have
  a `dcc.Store` payload >1MB. Today nothing comes close.
- **AG Grid `rowModelType="serverSide"`** — adopt when a single grid
  exceeds 10k rows. Trades grids might cross this bar in production;
  flag it then.

---

## 7. Build workflow

A new page is built in five sequential PRs. Each is independently
reviewable and revertible. Mixing them is the cause of the "Trade Graph
was wrong from day one" failure mode we just lived through.

> **Shortcut:** [`page_template/`](./page_template/) provides a copy-paste
> layout + callback pair that already implements PRs 1-3 below.  Use it
> as your PR-1 starting point — it bakes in the mount-tripwire,
> capture/render split, pathname gates, `figure_with_fallback`, and
> uirevision keying so you don't re-derive any of those.

### PR 1 — Static layout from mock

`build_<page>(mock_session)` renders the entire page tree with hard-coded
sample data. **No callbacks registered.** Run the app, navigate to the
page, eyeball it side-by-side with the design mock, fix CSS, ship.

The mock data lives at the top of the layout module, prefixed `_MOCK_`,
and is the *only* data the page sees in this PR. Callbacks come later.

Definition of done: page renders without errors, matches the mock image
to within agreed tolerance.

### PR 2 — Capture callbacks

Wire user actions → `session-store.data`. Each capture callback has a
single output (the session store) and a small input set. Run the app,
edit a filter, refresh the tab, confirm the filter survived (proves
session round-trip works).

Definition of done: every interactive control writes to session and the
session-write survives a tab refresh.

### PR 3 — Render callbacks

Wire `session.data` → UI props. Render callbacks now drive the figures,
grids and KPI cards that PR 1 hard-coded. Mock data factory in the
layout module is replaced by a `_render_<page>(session)` callback in
the cb module.

Definition of done: changing a filter changes the chart / grid / KPI
without a tab refresh.

### PR 4 — Live data

Replace the mock data factory with `backend.<method>_df(...)` calls.
Add the four-state UI handling (loading / data / empty / error) per
§5.1.

Definition of done: page works against the real FastAPI server with
real artefacts.

### PR 5 — Performance pass

Apply P1 / P2 / P3 levers where they help. Run the page with browser
dev tools open, identify any callback >100ms, decide between patch /
clientside / leave-alone. Often a no-op PR.

Definition of done: no avoidable >100ms callbacks; user zoom survives
data refresh.

---

## 8. Worked example — Evaluation filter bar refactor

This is the page-contract-compliant version of the bar we've been
struggling with. Compare to current `callbacks/evaluation_cb.py`.

### Layout

```python
# layouts/evaluation/shell.py
def build_evaluation(*, session: Session, pathname: str) -> html.Div:
    active = active_subtab_from_path(pathname)
    return html.Div(
        id=EVALUATION_IDS["root"],
        className="rade-page rade-evaluation",
        children=[
            build_evaluation_filter_bar(            # ← session-driven
                initial_filters=session.evaluation.filters,
                initial_open=session.evaluation.filter_bar_open,
            ),
            _tabs_row(active_slug=active),
            html.Div(
                id=EVALUATION_IDS["content"],
                children=build_subtab_content(active, session=session),
            ),
        ],
    )
```

`build_evaluation_filter_bar` accepts `initial_filters` and threads
`value=...` into every `MultiSelect` and the `DatePickerInput`. Chips
and the "n active" label are also computed from `initial_filters` at
build time, so the page is fully usable before any callback fires.

### Capture callbacks (writes to session only)

```python
# callbacks/evaluation_cb.py — _register_capture
@app.callback(
    Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
    Input(EVAL_FILTER_IDS["asset_class"], "value"),
    Input(EVAL_FILTER_IDS["currency"],    "value"),
    Input(EVAL_FILTER_IDS["desk"],        "value"),
    Input(EVAL_FILTER_IDS["product"],     "value"),
    Input(EVAL_FILTER_IDS["date_range"],  "value"),
    State(SHELL_IDS["session_store"],     "data"),
    State(SHELL_IDS["url"],               "pathname"),
    prevent_initial_call=True,
)
def _capture_filters(asset_class, currency, desk, product, date_range,
                     session_data, pathname):
    if not pathname or not pathname.startswith("/evaluation"):  # Rule C4
        raise PreventUpdate
    session = Session.from_store(session_data)
    df, dt = _parse_date_range(date_range)
    session.evaluation.filters = EvaluationFilters(
        asset_class=list(asset_class or []),
        currency=list(currency or []),
        desk=list(desk or []),
        product=list(product or []),
        date_from=df,
        date_to=dt,
    )
    return session.to_store()
```

```python
# Drawer toggle — captures the open/closed preference into session.
@app.callback(
    Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
    Output(EVAL_FILTER_IDS["collapse"], "opened"),
    Input(EVAL_FILTER_IDS["toggle_btn"], "n_clicks"),
    State(EVAL_FILTER_IDS["collapse"], "opened"),
    State(SHELL_IDS["session_store"], "data"),
    prevent_initial_call=True,
)
def _capture_drawer(n, currently_open, session_data):
    if not n: raise PreventUpdate
    new_open = not bool(currently_open)
    session = Session.from_store(session_data)
    session.evaluation.filter_bar_open = new_open
    return session.to_store(), new_open
```

```python
# Clear-all / Reset / chip × — write empty filters to session,
# the layout's value props will be re-rendered by the render
# callbacks in the next section.  Three callbacks share a common
# helper that returns a session with zero filters.
```

### Render callbacks (clientside where trivial)

```javascript
// assets/js/evaluation.js
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.evaluation = {
    update_filter_ui: function(asset_class, currency, desk, product, dateRange) {
        // Returns [chips, toggleLabel, clearAllStyle] — pure function of
        // inputs already in the browser, zero server round-trip.
        ...
    },
};
```

```python
# callbacks/evaluation_cb.py — _register_render
app.clientside_callback(
    ClientsideFunction(namespace="evaluation",
                       function_name="update_filter_ui"),
    Output(EVAL_FILTER_IDS["chips"],         "children"),
    Output(EVAL_FILTER_IDS["toggle_label"],  "children"),
    Output(EVAL_FILTER_IDS["clear_all"],     "style"),
    Input(EVAL_FILTER_IDS["asset_class"], "value"),
    Input(EVAL_FILTER_IDS["currency"],    "value"),
    Input(EVAL_FILTER_IDS["desk"],        "value"),
    Input(EVAL_FILTER_IDS["product"],     "value"),
    Input(EVAL_FILTER_IDS["date_range"],  "value"),
)
```

Net effect:

- 4 callbacks → 1 capture per dimension cluster + 1 clientside render.
- No `allow_duplicate` on `Collapse.opened` (single writer).
- No hydration callback (initial values come from layout).
- No input→input chains (clientside reads inputs, capture writes session,
  no callback writes input `value`).
- `eval-filter-toggle-label` warning gone (toggle label is only written
  clientside, when its parent DOM is necessarily mounted).

---

## 9. Pre-flight checklist

A reviewer of any layout PR or callback PR should be able to walk this
list in 60 seconds.

### Layout PR

- [ ] Module exports `<PAGE>_IDS`, `build_<page>(...)`, `__all__`.
- [ ] `build_<page>` is a pure function (no side effects, no callback
      registration, no backend access).
- [ ] Initial values for every interactive control come from `session`
      (or a documented mock dict in PR 1).
- [ ] No hardcoded id strings outside `<PAGE>_IDS`.
- [ ] No inline `style={...}` with visual styling (className-driven).
- [ ] Page renders cleanly with `Session()` (defaults) and with a
      filled-in session.
- [ ] Side-by-side with the design mock — pixel-level match.

### Callback PR

- [ ] Module exports a single `register(app, backend)` function.
- [ ] Callbacks split into `_register_capture` and `_register_render`
      sections (or `_register_<feature>` if a section grows >5 callbacks).
- [ ] `prevent_initial_call` follows Rule C5's role-based table —
      `True` on capture, `"initial_duplicate"` on bootstrap + every
      mount-tripwire-driven render. **No `prevent_initial_call=False`
      on any page-scoped callback.**
- [ ] Every callback gates on `pathname` as **`State`** (Rule C4).
      `grep -nE 'Input\(\s*SHELL_IDS\["url"\]\s*,\s*"pathname"\)'
      <your_page>_cb.py` returns nothing.
- [ ] Every page-scoped render callback's first trigger Input is
      `Input(<PAGE>_IDS["mount_signal"], "data")` (Rule C7); session-store
      is the second Input.
- [ ] Layout root contains a `dcc.Store(id=<PAGE>_IDS["mount_signal"],
      data=True, storage_type="memory")` (Rule L4).
- [ ] Every `Output` has at most one writer, OR `allow_duplicate=True`
      with a one-line comment.
- [ ] No callback's `Output` is another callback's `Input` (Rule C3).
- [ ] Capture callbacks output only `session-store.data`.
- [ ] Render callbacks have only `mount_signal.data`,
      `session-store.data` and (optional) page-internal stores as
      `Input`s.
- [ ] Trivial UI derivations (counts, labels, classNames) are
      `clientside_callback`s.
- [ ] Backend results are tri-state-handled — every render callback
      consuming a `BackendResult` either calls `figure_with_fallback`
      / `component_with_fallback` (from `..data`) or branches the
      three states (`not res.ok` / `is_empty` / `ok`) explicitly.
- [ ] Time-series figures have `uirevision_key=...` keyed on the data
      domain (`split`, `cluster_id`, or a tuple).

### Code-style (already enforced de facto)

- [ ] `from __future__ import annotations` at module top.
- [ ] `TYPE_CHECKING` guard for `Dash` and `RadeBackend` imports.
- [ ] Module-private constants prefixed `_`, public ones not.
- [ ] Module-level `logger = logging.getLogger(__name__)` if logging.
- [ ] ASCII section dividers between logical groups
      (`# ── Capture ─────────`).
- [ ] `__all__` at module foot.

---

## 10. Anti-patterns catalogue

Specific things we've seen go wrong, with the fix.

### A1 — Hydration callback writes input `value`s after mount

**Symptom:** "nonexistent object was used in an Output" warning fires
during cross-page navigation.
**Cause:** chained capture callback fires before the new chrome's DOM
is committed.
**Fix:** Rule L1. Initial values from session at build time.
**Example:** `_hydrate_filter_bar` → killed in §8.

### A2 — `e.map is not a function` in console

**Symptom:** browser console error, page renders but interaction is
flaky.
**Cause:** AG Grid `columnDefs.type` was given a string. AG Grid's
typescript signature is `type?: string | string[]` and the JS runtime
does `.map(...)` over it — strings have no `.map`.
**Fix:** always use a list, even for a single type:
`type=["numericColumn"]`.

### A3 — `Patch()` returned for first render

**Symptom:** chart is empty on page load.
**Cause:** `dash.Patch()` is a *delta* relative to an existing figure;
the first render needs a full `go.Figure`.
**Fix:** branch on whether the figure has been rendered before
(`State("chart", "figure")` and check non-empty `data`).

### A4 — `prevent_initial_call` missing

**Symptom:** callback fires on every page load with `None` inputs;
either crashes or pollutes session with default values.
**Fix:** Rule C5. `prevent_initial_call=True` is mandatory.

### A5 — Session schema drift

**Symptom:** old browser tabs with stale `dcc.Store` data send
unexpected fields; field-renames silently fail.
**Fix:** existing pattern in `data/session.py` — bump
`SESSION_SCHEMA_VERSION` whenever you add/rename/remove a field,
`Session.from_store()` rejects mismatched versions.

### A6 — `dcc.Store` used as a cache for backend data

**Symptom:** large JSON in the browser, slow page loads, sometimes
hits Dash's payload size limit.
**Cause:** `RadeBackend` already memoizes via `flask_caching`. Storing
the same data in `dcc.Store` is double-caching with the worse cache.
**Fix:** call `backend.<method>_df(...)` from the render callback every
time. The cache makes it free.

### A7 — Topbar / sidebar callbacks defined under a page module

**Symptom:** page Y's callback module fires when the user is on page X.
**Cause:** topbar / sidebar live in `layouts/shell.py` — their
callbacks belong in `callbacks/shell_cb.py` (or co-located in
`callbacks/__init__.py`'s register fan-out), not in any page module.
**Fix:** module-scope discipline. A page callback module touches only
ids inside that page's subtree.

### A8 — Render callback triggers on `Input(pathname)` (mount race)

**Symptom:** the page renders the static layout (card chrome, KPI
labels, axis titles) but every dynamic body is empty — no `—`
placeholders, no values, no chart traces, no AG Grid rows. No
JavaScript console error, no server-side traceback, the callback
appears to never fire. The first sub-tab transition you make later
*does* paint correctly, leaving you guessing what changed.

**Cause:** the callback declares `Input(SHELL_IDS["url"], "pathname")`,
which fires the moment the URL changes. The router's `_sync_from_url`
fires on the same edge to swap the sub-tab content. The render
callback wins the race, writes to outputs whose IDs aren't in the DOM
yet, and Dash silently drops the writes. By the time the new layout
mounts, no Input has changed since, so the render never re-fires.

The trade-graph render callback survives this by accident: its
bootstrap *always* writes `session-store.data` on first hit, which
re-triggers the render via the `Input(session_store)` once outputs
exist. As soon as you have a code path where the bootstrap returns
`no_update` for `session-store` (e.g. canonical cluster already
matches the layout seed), the page silently breaks.

**Fix:** Rule C7. Trigger off `Input(<PAGE>_IDS["mount_signal"], "data")`
+ `Input(SHELL_IDS["session_store"], "data")`; demote `pathname` to a
`State`-side gate; set `prevent_initial_call="initial_duplicate"`. The
mount tripwire (Rule L4) lives inside the swapped layout chunk so it
can't race the swap.

---

## 11. Out of scope (today)

The following are deliberately *not* in this contract; flag a discussion
when we hit pain:

- Migrating to Dash's built-in `dash.register_page` Pages router.
  Today's custom router has chrome-rebuild semantics we'd have to
  re-implement; not worth the churn.
- URL-encoded filter state for shareable links. Useful eventually; not
  blocking.
- A11y testing harness. Pages are tagged with `aria-*` attributes
  already; formal audit is a Phase G task.
- Background callbacks / disk-cached long-running fetches. Add when
  needed.

---

## 12. Change log

- v1 (this doc) — initial draft. Codifies existing structure (layouts /
  callbacks / components / figures / data / Session / RadeBackend)
  + adds 6 callback rules + 5-step build workflow + 3 performance levers
  + 7 anti-patterns we've actually hit.
- v1.1 — adds `figure_with_fallback` / `component_with_fallback`
  (`data/result_helpers.py`) for tri-state rendering, and the page
  template scaffold under [`page_template/`](./page_template/).
  No rule changes; the helpers + template just operationalise the
  rules so adding a new page is a 15-minute copy-paste.
- v1.2 — adds **Rule L4** (mount tripwire is mandatory),
  **Rule C7** (page-scoped renders trigger on `mount_signal`, not
  `pathname`), strengthens **Rule C4** to forbid `Input(pathname)` on
  page-scoped callbacks, rewrites **Rule C5** to enumerate legal
  `prevent_initial_call` values per callback role, and adds
  **Anti-pattern A8** (the mount race). Closes the gap that let
  Cluster Deep-Dive's render callbacks paint empty on first visit:
  the template was using `Input(pathname)` + `prevent_initial_call=False`,
  in conflict with Rule C4. Template files (`template_cb.py`,
  `template_layout.py`, `README.md`) updated in the same revision.
