# Rade UI Design Spec

Short, living spec for the Rade Dash UI. The **mock images in this folder
(`rade_*.png`) are the design contract** — every new tab, card or callback
must visually and functionally align with one of them. When the mocks and
the code disagree, update one of them deliberately, never silently.

---

## 1. Brand

| Token | Value |
|---|---|
| Name | **Rade** |
| Tagline | Quantitative Model Intelligence |
| Logomark | Violet → cyan geometric "R" prism |
| Wordmark | Inter / Geist, weight 600, tracking -0.02em |
| Tone | Quietly confident, low-ornament, numerically dense. Linear / Vercel / Raycast school. |

---

## 2. Palette

Dark-only for v1. All tokens are Tailwind defaults — no bespoke colours.

| Role | Tailwind | Hex |
|---|---|---|
| Background | `slate-950` | `#020617` |
| Surface / card | `slate-900` | `#0f172a` |
| Sunken / track | `slate-950` inside card | `#020617` |
| Border | `slate-800` | `#1e293b` |
| Border subtle | `slate-800/60` | rgba |
| Text primary | `slate-100` | `#f1f5f9` |
| Text secondary | `slate-400` | `#94a3b8` |
| Text muted | `slate-500` | `#64748b` |
| Brand primary | `violet-500` | `#8b5cf6` |
| Brand accent | `cyan-400` | `#22d3ee` |
| Brand gradient | `from-violet-500 to-cyan-400` | — |
| Success | `emerald-500` | `#10b981` |
| Warning | `amber-500` | `#f59e0b` |
| Danger | `rose-500` | `#f43f5e` |
| Info | `sky-400` | `#38bdf8` |

**Rule:** Brand gradient is reserved for primary CTAs, selected nav
indicators, and the splash logomark. Do not apply it to card backgrounds
or chart fills — use 20–30% opacity of the violet for gradient chart fills.

---

## 3. Typography

- **Font**: Inter (fallback: system). Geist is acceptable for brand assets only.
- **Numeric font**: `JetBrains Mono` / `IBM Plex Mono` in tables and diff views
  (Tailwind `font-mono`).
- **Scale**:

| Usage | Size | Weight | Tracking |
|---|---|---|---|
| Page title | 28–36px | 600 | -0.01em |
| Section title | 18–20px | 600 | 0 |
| Card title | 14–15px | 600 | 0 |
| Body | 13–14px | 400 | 0 |
| Muted / caption | 11–12px | 400 | 0.01em |
| Code / numeric | 13px mono | 400 | 0 |

---

## 4. Layout grid

- **Viewport floor**: 1280px. Responsive down to 1024px (sidebar collapses).
- **Sidebar**: 220px wide, `bg-slate-900`, `border-r border-slate-800`.
- **Content padding**: `px-8 py-6` on the main area.
- **Card padding**: `p-5` (default), `p-4` (compact tiles).
- **Gap**: `gap-4` within a card, `gap-6` between cards.
- **Rounded**: `rounded-2xl` on cards, `rounded-xl` on inputs, `rounded-lg` on
  small chips.
- **Shadow**: none by default, `shadow-sm` max. No glow on real content.
  Glow is reserved for: splash logo, selected graph node, ai-assistant
  accent ring.

---

## 5. Page skeleton

Every tab follows the same top-down rhythm — see `rade_landing_dashboard.png`
and `rade_eval_portfolio.png`:

1. **Breadcrumb** (slate-400, 12–13px) — `Section › Sub-section › Tab`.
2. **Title row** — H1 + subtitle + right-aligned meta (version chip, bell,
   search).
3. **Filter bar** (optional) — on a `bg-slate-900 rounded-2xl` strip.
4. **KPI tile row** — 3–6 equal-width cards.
5. **Main content grid** — 1–2–3 column layouts; charts left-weighted, tables
   right-weighted.
6. **Footer strip** — source artifact caption, export actions.

---

## 6. Component library

| Concept | Dash component | Notes |
|---|---|---|
| App shell | `dash-mantine-components.AppShell` | `padding="md"`, custom sidebar. |
| Navigation | `html.A` + Tailwind | Selected = `bg-slate-800` + `border-l-2 border-violet-500`. |
| Breadcrumb | `html.Div` + chevron icons | `dash-iconify` for icons. |
| Segmented control | `dmc.SegmentedControl` | Default style, brand color on active. |
| Dropdown | `dcc.Dropdown` | Theme overridden via `className`. |
| Multi-select | `dmc.MultiSelect` | For cluster/trade pickers. |
| Range slider | `dmc.RangeSlider` | For scenario windows. |
| Button primary | `dmc.Button` with gradient | Variant `gradient`, from violet-500 to cyan-400. |
| Button outline | `dmc.Button variant="outline"` | |
| Pill / chip | `dmc.Badge` | With per-status color. |
| Metric tile | `html.Div` + Tailwind | `rounded-2xl bg-slate-900 border border-slate-800 p-5`. |
| Chart | `dcc.Graph` (Plotly) | Shared dark template — see §7. |
| Table | `dash-ag-grid` | Shared theme config — see §8. |
| Network graph | `dash-cytoscape` | Layout `cose-bilkent`. |
| File upload | `dcc.Upload` | Used in Inference Console. |
| Timeline | `dmc.Timeline` | Governance lineage. |
| Drag canvas | `dash-draggable` | Report Builder only. |
| Command palette | `dmc.Spotlight` | Cmd+K globally. |
| Modal | `dmc.Modal` | Approvals, confirmations. |
| Skeleton | `dmc.Skeleton` | Loading state for every card. |
| Alert | `dmc.Alert` | Error / warning banners. |
| Toast | `dmc.Notification` | Save/export confirmations. |

---

## 7. Chart defaults (Plotly)

One shared template `RADE_DARK` applied via `go.Figure(layout=...)` or a
helper `rade_figure()`:

- `template="plotly_dark"` as base, then override:
- `paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"` (transparent; card shows through).
- `font=dict(family="Inter, sans-serif", size=12, color="#94a3b8")`.
- Gridlines: `gridcolor="rgba(148,163,184,0.1)"`, `zerolinecolor` same.
- Axis ticks: `tickcolor="rgba(148,163,184,0.2)"`, `ticks="outside"`, `tickfont.color="#94a3b8"`.
- Margin: `dict(l=40, r=20, t=30, b=40)`.
- Legend: top-right, horizontal, chip-style (`bgcolor="rgba(15,23,42,0.6)"`, `bordercolor="#1e293b"`).
- Hovermode: `"x unified"` for time series, `"closest"` otherwise.
- **Primary series**: `violet-500`. **Secondary**: `cyan-400`. **Reference /
  actual**: dashed `slate-400`. **Residual band**: `rgba(139,92,246,0.18)` fill.
- Colormaps:
  - Residual heatmap: `[[0,"#7c3aed"],[0.5,"#334155"],[1,"#f43f5e"]]`.
  - Correlation heatmap: `"RdBu_r"` centred at 0.
  - Completeness heatmap: `[[0,"#f43f5e"],[0.5,"#334155"],[1,"#7c3aed"]]`.
  - Sequential: Plotly `Viridis` (keep for simple magnitudes).

All chart code lives under `ui/components/charts/` so the template is
enforced centrally.

---

## 8. Table defaults (`dash-ag-grid`)

Single shared `RADE_GRID_DEFAULTS` dict:

```python
RADE_GRID_DEFAULTS = dict(
    className="ag-theme-alpine-dark rade-grid",
    defaultColDef={
        "sortable": True,
        "filter": True,
        "resizable": True,
        "cellClass": "text-slate-200",
    },
    dashGridOptions={
        "rowHeight": 36,
        "headerHeight": 40,
        "suppressMovableColumns": False,
        "animateRows": True,
    },
)
```

- Numeric columns: `type="numericColumn"`, right-aligned, `font-mono`.
- Status pills: `cellRenderer` returning `dmc.Badge`.
- Inline bars: custom `cellRenderer` using a Tailwind-styled `<div>` with
  width proportional to value; bar colour follows status (`bg-rose-500/30`
  for breaches, `bg-emerald-500/30` for healthy).
- Row stripes: `ag-row-odd` = `bg-slate-950/40`.

---

## 9. State reference

Every card must handle the 6 states shown in `rade_states_reference.png`:

1. **Loading** — `dmc.Skeleton` matching the final layout's shape.
2. **Empty (no data)** — outlined icon + heading + body + primary action.
3. **Empty (filter too narrow)** — funnel icon + "Reset filters" link.
4. **Error (404 artifact)** — specific, names the missing file, lists
   available alternatives.
5. **Error (backend down)** — cites the `/health` status, offers Retry.
6. **Authz denied** — names the restricted resource, offers access request.

Do not ship a card without at least the Loading and one Empty state.

---

## 10. Accessibility

- All interactive elements reachable by keyboard; focus ring =
  `ring-2 ring-violet-500 ring-offset-2 ring-offset-slate-950`.
- Colour contrast ≥ 4.5:1 for body text against `slate-900` (checked: all
  palette entries pass).
- Charts ship an accompanying data table under a "View raw data" disclosure
  for screen-reader access.
- Status colour is never the only signal — pair with icon + label.
- Cmd+K command palette exposes every navigable destination.

---

## 11. Design contract — the 20 mocks

| # | File | Purpose | Data source(s) |
|---|---|---|---|
| 1 | `rade_splash.png` | Boot / loading | `/health`, `/versions` |
| 2 | `rade_login.png` | SSO sign-in | — |
| 3 | `rade_landing_dashboard.png` | Home / overview | `/overview`, `/metrics/ensemble`, `/portfolio`, `/clusters` |
| 4 | `rade_eval_portfolio.png` | Evaluation — portfolio split | `/portfolio`, `/metrics/ensemble` |
| 5 | `rade_eval_by_desk.png` | Evaluation grouped by attribute | `/metrics/per-member`, `/clusters` |
| 6 | `rade_cluster_deep_dive.png` | Per-cluster drill-down | `/clusters`, `/metrics/per-member`, `/cluster-timeseries`, `/trades`, `/graph-stats` |
| 7 | `rade_cross_cluster.png` | Correlation matrix + group stats | `/group-correlations`, `/clusters` |
| 8 | `rade_trade_graph.png` | Network graph | `/graph-stats`, `/trades` |
| 9 | `rade_data_quality.png` | Completeness + feature summary | `/quality/completeness`, `/quality/feature-summary` |
| 10 | `rade_model_monitoring.png` | Live drift + alerts | new `/monitoring/*` (future) |
| 11 | `rade_training_curves.png` | Per-cluster training traces | `training_curves.parquet` (served via `/training/curves`) |
| 12 | `rade_prediction_explorer.png` | Raw trade × scenario NPZ drill-in | `/predictions`, `/trades` |
| 13 | `rade_governance.png` | Registry + lineage + approvals | `/versions`, new `/governance/*` |
| 14 | `rade_version_comparison.png` | Side-by-side diff of two versions | compose any `/metrics/*` + `/portfolio` for two versions |
| 15 | `rade_inference.png` | Run ensemble on new trades | new `/inference/run` (future) |
| 16 | `rade_scenario_lab.png` | Synthetic scenario builder + score | new `/scenarios/*` (future) |
| 17 | `rade_report_builder.png` | Block-canvas report authoring | reads anything above, emits PDF / HTML |
| 18 | `rade_ai_assistant.png` | Anomaly investigation side-panel | LLM backend + tool calls into the rest of the API |
| 19 | `rade_command_palette.png` | Global cmd+K navigation | static + `/overview` |
| 20 | `rade_states_reference.png` | Loading / empty / error atlas | — |

---

## 12. Extension process

When a new tab or card is proposed:

1. Draft or regenerate a mock under `docs/platform_designs/rade_<name>.png`.
2. Append a row to §11 table naming its data source and required endpoints.
3. Check charts/tables against §7 / §8 defaults — add to shared helpers if
   a new pattern emerges.
4. Add state coverage (§9) to the implementation PR checklist.

No mock, no merge.



---

## Appendix A — `layouts/splash.py` (temporary)

Paste this verbatim into `src/ui/apps/rade_analytics/layouts/splash.py`, replacing the current contents. This appendix will be removed once verified on the work env.

Line count: 221.

```python
"""Splash page — the "front door" before users enter the main app.

Purpose
-------
* Announce the product identity (logo + wordmark + tagline).
* Show API liveness at a glance (status dot + URL).
* Let the user confirm / switch which ensemble version they want to
  browse before landing on the overview.
* One CTA: "Enter Rade" → writes the chosen version into the session
  store and navigates to ``/`` (overview).

This module only owns layout.  Data-fetching and navigation are wired
by :mod:`..callbacks.splash_cb`, which targets the DOM ids exported in
:data:`SPLASH_IDS` so callbacks never hardcode strings.

Styling contract
----------------
All visual treatment comes from the ``rade-splash-*`` classes defined
at the bottom of ``assets/rade.css``.  The layout deliberately avoids
Tailwind arbitrary values (``min-h-[calc(...)]``, ``w-[28rem]``, etc.)
because the repository ships a pre-compiled ``rade.css`` — new
arbitrary values would silently no-op without a Tailwind rebuild.  The
custom classes give us stable visuals with zero build step.

Design spec anchors
-------------------
* §2 Palette — gradient glows mirror the logo gradient
  (violet-500 → cyan-400).
* §3 Typography — Inter tagline, JetBrains Mono for version ids.
* §6 Components — DMC Select + Button.
"""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify

from ..components.brand import Brand


# Every dynamic id on the splash lives here so callbacks in
# splash_cb.py never hardcode strings.  Also used by tests to target
# individual elements.
SPLASH_IDS = {
    "root":              "splash-root",
    "status_dot":        "splash-status-dot",
    "status_label":      "splash-status-label",
    "api_url":           "splash-api-url",
    "active_version":    "splash-active-version",
    "version_select":    "splash-version-select",
    "enter_btn":         "splash-enter-btn",
    "error_banner":      "splash-error-banner",
}


def build_splash() -> html.Div:
    """Build the splash page layout.

    The layout renders in an "empty" state (no version data, no
    confirmed API liveness) — the bootstrap callback in
    :mod:`..callbacks.splash_cb` populates all dynamic pieces on mount.
    """
    return html.Div(
        id=SPLASH_IDS["root"],
        className="rade-splash-root",
        children=[
            _backdrop(),
            _card(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Backdrop — soft gradient glow behind the card
# ─────────────────────────────────────────────────────────────────────


def _backdrop() -> html.Div:
    """Decorative blurred gradient so the card reads against depth."""
    return html.Div(
        className="rade-splash-backdrop",
        children=[
            html.Div(className="rade-splash-glow-violet"),
            html.Div(className="rade-splash-glow-cyan"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Card — the interactive surface (brand, status, picker, CTA)
# ─────────────────────────────────────────────────────────────────────


def _card() -> html.Div:
    return html.Div(
        className="rade-splash-card",
        children=[
            _brand_block(),
            _status_strip(),
            _version_block(),
            _cta_block(),
            _footer(),
        ],
    )


def _brand_block() -> html.Div:
    return html.Div(
        className="rade-splash-brand-block",
        children=[
            Brand(size="xl", show_wordmark=True),
            html.Div(
                "Ensemble analytics for institutional trading.",
                className="rade-splash-tagline",
            ),
        ],
    )


def _status_strip() -> html.Div:
    """Live / offline indicator + the API URL the app is pointed at.

    Rendered in a neutral ("Connecting...") state; the bootstrap
    callback flips the dot colour and label once ``backend.health()``
    resolves.
    """
    return html.Div(
        className="rade-splash-status-strip",
        children=[
            html.Div(
                className="rade-splash-status-inner",
                children=[
                    html.Div(
                        id=SPLASH_IDS["status_dot"],
                        className="rade-status-dot rade-status-dot--booting",
                    ),
                    html.Span(
                        "Connecting...",
                        id=SPLASH_IDS["status_label"],
                        className="rade-splash-status-label",
                    ),
                ],
            ),
            html.Span(
                "—",
                id=SPLASH_IDS["api_url"],
                className="rade-splash-api-url",
            ),
        ],
    )


def _version_block() -> html.Div:
    """Active-version headline + Select to switch before entering."""
    return html.Div(
        className="rade-splash-version-block",
        children=[
            html.Div(
                className="rade-splash-version-inner",
                children=[
                    html.Div(
                        "ACTIVE VERSION",
                        className="rade-section-label",
                    ),
                    html.Div(
                        "—",
                        id=SPLASH_IDS["active_version"],
                        className="rade-splash-version-headline",
                    ),
                ],
            ),
            dmc.Select(
                id=SPLASH_IDS["version_select"],
                label="Switch version",
                placeholder="Loading available versions...",
                data=[],
                value=None,
                clearable=False,
                searchable=True,
                w="100%",
                leftSection=DashIconify(icon="tabler:git-branch", width=14),
                disabled=True,
            ),
        ],
    )


def _cta_block() -> html.Div:
    """Primary Enter button + fallback error banner for bootstrap errors."""
    return html.Div(
        className="rade-splash-cta-block",
        children=[
            html.Div(
                id=SPLASH_IDS["error_banner"],
                className="rade-splash-error-banner",
                children="",
            ),
            dmc.Button(
                id=SPLASH_IDS["enter_btn"],
                children="Enter Rade",
                rightSection=DashIconify(icon="tabler:arrow-right", width=16),
                size="md",
                fullWidth=True,
                variant="gradient",
                gradient={"from": "violet", "to": "cyan", "deg": 135},
                disabled=True,
            ),
        ],
    )


def _footer() -> html.Div:
    return html.Div(
        "Rade UI · v0.1",
        className="rade-splash-footer",
    )


__all__ = ["SPLASH_IDS", "build_splash"]
```
