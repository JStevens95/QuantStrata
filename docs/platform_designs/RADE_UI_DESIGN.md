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

## Appendix A — `assets/logo.svg` (temporary)

Full replace. Final "Sliced R" mark chosen from round-one logo review — bold geometric R with a violet→cyan diagonal gradient fill and a single diagonal slice through the lower leg. The slice is cut via an SVG mask, so the gap reveals whatever sits behind the logo — works equally well on slate-950, on light surfaces, or on the gradient backdrop itself.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!--
  Rade Analytics - primary brand mark ("Sliced R").

  Bold geometric capital R filled with a violet -> cyan diagonal
  linear gradient, with a single thin diagonal slice cutting through
  the lower leg.

  The slice is rendered as an SVG mask (not a hard-coded coloured
  polygon on top) so the mark stays background-agnostic: the gap
  reveals whatever sits behind the logo.  Works on slate-950, on
  light surfaces, on the violet-to-cyan gradient itself - the cut
  always reads correctly.

  viewBox 0 0 100 120 is the native aspect for the capital R; the
  mark scales crisply from 16 x 19 px (favicon) to 144 x 173 px
  (splash hero) without hinting artefacts.

  Colour tokens mirror RADE_UI_DESIGN.md section 2:
    - violet-500 (#8b5cf6)  gradient stop 0%
    - cyan-400   (#22d3ee)  gradient stop 100%
-->
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 100 120"
     fill="none"
     role="img"
     aria-label="Rade logo">
  <title>Rade</title>

  <defs>
    <linearGradient id="rade-body-gradient"
                    x1="0" y1="0" x2="100" y2="120"
                    gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#22d3ee"/>
    </linearGradient>

    <!--
      Diagonal slice, 4 px thick, angled ~16 degrees above horizontal.
      Everything inside the black polygon is cut away from the R; the
      white rect is the "kept" region.  Extends slightly past the
      right edge so the cut exits cleanly through the leg's outer
      edge.
    -->
    <mask id="rade-slice-mask"
          maskUnits="userSpaceOnUse"
          x="0" y="0" width="100" height="120">
      <rect x="0" y="0" width="100" height="120" fill="white"/>
      <polygon points="20,92 96,70 96,74 20,96" fill="black"/>
    </mask>
  </defs>

  <!--
    Single path for the whole R letterform:
      * Outer sub-path (clockwise):
          stem top -> bowl top -> bowl outer curve ->
          leg outer diagonal -> leg bottom ->
          leg inner diagonal (up-left to the stem/leg junction) ->
          stem right edge (down) -> stem bottom -> close
      * Inner sub-path: rounded counter inside the bowl, cut out via
        the evenodd fill-rule.
  -->
  <path d="M 10 10
           L 60 10
           Q 82 10 82 32
           Q 82 54 60 54
           L 90 108
           L 58 108
           L 34 54
           L 34 108
           L 10 108
           Z
           M 40 22
           L 56 22
           Q 68 22 68 32
           Q 68 42 56 42
           L 40 42
           Z"
        fill="url(#rade-body-gradient)"
        fill-rule="evenodd"
        mask="url(#rade-slice-mask)"/>
</svg>
```

---

## Appendix B — `assets/rade.css` (append only)

**Do not replace the whole file** — scroll to the very bottom of `rade.css` (after the `@keyframes rade-status-pulse` block) and **append** the block below. Adds the splash-v2 hero-outside-card classes while leaving the original splash block intact.

File should go from 808 lines to 903 lines after paste.

```css

/* ─────────────────────────────────────────────────────────────────── */
/* SPLASH PAGE v2 — hero outside card + top-right version pill.        */
/* Added on top of the original splash block; older classes remain in  */
/* case any other page reuses them.                                    */
/* ─────────────────────────────────────────────────────────────────── */

.rade-splash-layout {
  position: relative;
  z-index: 10;
  width: 100%;
  max-width: 72rem;
  min-height: 100vh;
  margin: 0 auto;
  padding: 1.5rem 2rem 2rem 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.rade-splash-topbar {
  width: 100%;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  min-height: 2rem;
}

.rade-splash-version-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.75rem;
  color: #cbd5e1;
  background-color: rgb(30 41 59 / 0.6);
  border: 1px solid #334155;
  backdrop-filter: blur(4px);
}

.rade-splash-center {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3rem;
  width: 100%;
}

.rade-splash-hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.25rem;
  text-align: center;
}

.rade-splash-hero-logo {
  width: 9rem;
  height: 12rem;
  filter: drop-shadow(0 14px 28px rgb(139 92 246 / 0.35));
}

.rade-splash-hero-title {
  font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
  font-size: 5rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1;
  color: #f8fafc;
  margin: 0;
}

.rade-splash-hero-subtitle {
  font-size: 0.75rem;
  font-weight: 500;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: #64748b;
  margin: 0;
}

.rade-splash-copyright {
  position: absolute;
  bottom: 1.25rem;
  left: 2rem;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.7rem;
  color: #475569;
  letter-spacing: 0.05em;
  z-index: 10;
}
```

---

## Appendix C — `layouts/splash.py` (temporary)

Full replace.

Line count:      254.

```python
"""Splash page — the "front door" before users enter the main app.

Layout
------
Full-viewport flex column:

    top-right          : active-version pill
    centre (flex-grow) : hero (logo + RADE + tagline) stacked over card
    bottom-left        : small copyright / build line

Only interactive elements live in the card (status strip, version
Select, Enter CTA, error banner); the identity block (logo +
wordmark + tagline) deliberately sits above the card so it reads
against the gradient backdrop rather than being boxed in.

Purpose
-------
* Announce the product identity.
* Show API liveness at a glance (status dot + URL).
* Let the user confirm / switch which ensemble version they want to
  browse before landing on the overview.
* One CTA: "Enter Rade" -> writes the chosen version into the session
  store and navigates to ``/`` (overview).

This module only owns layout.  Data-fetching and navigation are wired
by :mod:`..callbacks.splash_cb`, which targets the DOM ids exported in
:data:`SPLASH_IDS` so callbacks never hardcode strings.

Styling contract
----------------
All visual treatment comes from the ``rade-splash-*`` classes defined
in ``assets/rade.css``.  The layout deliberately avoids Tailwind
arbitrary values because the repository ships a pre-compiled
``rade.css`` — new arbitrary values would silently no-op without a
Tailwind rebuild.  The custom classes give us stable visuals with
zero build step.

Design spec anchors
-------------------
* §2 Palette — gradient glows mirror the logo gradient
  (violet-500 -> cyan-400).
* §3 Typography — Inter wordmark, JetBrains Mono for version ids.
* §6 Components — DMC Select + Button.
"""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


# Every dynamic id on the splash lives here so callbacks in
# splash_cb.py never hardcode strings.  Also used by tests to target
# individual elements.
#
# ``active_version`` now targets the top-right pill; the in-card
# headline has been removed since it duplicated the same value.
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
    """Build the splash page layout."""
    return html.Div(
        id=SPLASH_IDS["root"],
        className="rade-splash-root",
        children=[
            _backdrop(),
            html.Div(
                className="rade-splash-layout",
                children=[
                    _topbar(),
                    _center(),
                ],
            ),
            _copyright(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Backdrop — soft gradient glow behind everything
# ─────────────────────────────────────────────────────────────────────


def _backdrop() -> html.Div:
    """Decorative blurred gradient so the page reads against depth."""
    return html.Div(
        className="rade-splash-backdrop",
        children=[
            html.Div(className="rade-splash-glow-violet"),
            html.Div(className="rade-splash-glow-cyan"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Top bar — active version pill (flush right)
# ─────────────────────────────────────────────────────────────────────


def _topbar() -> html.Div:
    return html.Div(
        className="rade-splash-topbar",
        children=[
            html.Span(
                "-",
                id=SPLASH_IDS["active_version"],
                className="rade-splash-version-pill",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Centre column — hero stacked on top of interactive card
# ─────────────────────────────────────────────────────────────────────


def _center() -> html.Div:
    return html.Div(
        className="rade-splash-center",
        children=[
            _hero(),
            _card(),
        ],
    )


def _hero() -> html.Div:
    """Identity block: logo + RADE wordmark + small uppercase tagline."""
    return html.Div(
        className="rade-splash-hero",
        children=[
            html.Img(
                src="/assets/logo.svg",
                className="rade-splash-hero-logo",
                alt="Rade logo",
            ),
            html.H1("RADE", className="rade-splash-hero-title"),
            html.P(
                "Quantitative Model Intelligence",
                className="rade-splash-hero-subtitle",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Card — interactive surface (status, version switcher, CTA)
# ─────────────────────────────────────────────────────────────────────


def _card() -> html.Div:
    return html.Div(
        className="rade-splash-card",
        children=[
            _status_strip(),
            _version_switcher(),
            _cta_block(),
        ],
    )


def _status_strip() -> html.Div:
    """Live / offline indicator + the API URL the app is pointed at."""
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
                "-",
                id=SPLASH_IDS["api_url"],
                className="rade-splash-api-url",
            ),
        ],
    )


def _version_switcher() -> dmc.Select:
    """Version picker so users can browse an older ensemble before entering."""
    return dmc.Select(
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


# ─────────────────────────────────────────────────────────────────────
# Footer — bottom-left build / copyright line
# ─────────────────────────────────────────────────────────────────────


def _copyright() -> html.Div:
    return html.Div(
        "(c) Rade Platform - v0.1",
        className="rade-splash-copyright",
    )


__all__ = ["SPLASH_IDS", "build_splash"]
```


## Appendix D — `assets/rade.css` Phase-D append (temporary)

> **How to apply:** paste this block at the **end** of your work-env
> `src/ui/apps/rade_analytics/assets/rade.css`.  Nothing above it
> changes — this is 100% additive.  After confirming the overview page
> renders correctly, strip this appendix from `RADE_UI_DESIGN.md`.

```css

/* ==================================================================
 * OVERVIEW PAGE + TOPBAR v1  (Phase D)
 *
 * Classes referenced by:
 *   - components/topbar.py    (breadcrumb row)
 *   - layouts/overview.py     (page container, cluster heatmap,
 *                              feed/attention cards, quick actions)
 *
 * Kept in this single append-only block so the hand-authored rules
 * never collide with the pre-compiled Tailwind utilities above.
 * ================================================================== */

/* ── Topbar breadcrumb ─────────────────────────────────────────── */

.rade-breadcrumb {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  line-height: 1.25rem;
}

.rade-breadcrumb-item {
  color: #94a3b8;                     /* slate-400 */
  font-weight: 500;
  transition: color 120ms ease;
}

.rade-breadcrumb-item--active {
  color: #e2e8f0;                     /* slate-200 */
  font-weight: 600;
}

.rade-breadcrumb-separator {
  color: #475569;                     /* slate-600 */
}

/* ── Overview page container ───────────────────────────────────── */

.rade-page {
  padding: 1.5rem 1.75rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.rade-page-row {
  display: grid;
  gap: 1rem;
}

/* ── Cluster health heatmap (5x4 grid of coloured cells) ───────── */

.rade-cluster-heatmap {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.5rem;
  padding: 0.25rem 0;
}

.rade-heatmap-cell {
  aspect-ratio: 1 / 1;
  border-radius: 0.375rem;
  transition: transform 120ms ease, box-shadow 120ms ease;
  cursor: pointer;
  min-height: 40px;
}

.rade-heatmap-cell:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.35);
}

.rade-heatmap-cell--ok {
  background-color: #10b981;          /* emerald-500 */
}

.rade-heatmap-cell--warn {
  background-color: #f59e0b;          /* amber-500 */
}

.rade-heatmap-cell--err {
  background-color: #f43f5e;          /* rose-500 */
}

.rade-heatmap-cell--muted {
  background-color: #334155;          /* slate-700 — "no data" */
}

/* ── Feed / attention list rows (inside rade-card) ─────────────── */

.rade-list-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: #e2e8f0;                     /* slate-200 */
  margin-bottom: 0.5rem;
}

.rade-list-header {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  font-size: 0.7rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: #64748b;                     /* slate-500 */
  border-bottom: 1px solid #1e293b;   /* slate-800 */
}

.rade-list-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.625rem 0;
  font-size: 0.8125rem;
  color: #cbd5e1;                     /* slate-300 */
  border-bottom: 1px solid rgba(30, 41, 59, 0.5);
}

.rade-list-row:last-child {
  border-bottom: 0;
}

.rade-list-row-label {
  color: #e2e8f0;                     /* slate-200 */
  font-weight: 500;
}

.rade-list-row-sub {
  color: #94a3b8;                     /* slate-400 */
  font-size: 0.75rem;
  font-family: "JetBrains Mono", ui-monospace, monospace;
}

/* Emerald / rose text for +ve / -ve deltas in list rows. */
.rade-delta-pos { color: #34d399; font-family: "JetBrains Mono", ui-monospace, monospace; }
.rade-delta-neg { color: #fb7185; font-family: "JetBrains Mono", ui-monospace, monospace; }

/* Status chip used by "Attention Required" rows. */
.rade-chip {
  display: inline-flex;
  align-items: center;
  padding: 0.125rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: capitalize;
}

.rade-chip--amber {
  background-color: rgba(245, 158, 11, 0.15);
  color: #fbbf24;                     /* amber-400 */
  border: 1px solid rgba(245, 158, 11, 0.35);
}

.rade-chip--flagged {
  background-color: rgba(244, 63, 94, 0.15);
  color: #fb7185;                     /* rose-400 */
  border: 1px solid rgba(244, 63, 94, 0.35);
}

.rade-chip--ok {
  background-color: rgba(16, 185, 129, 0.15);
  color: #34d399;                     /* emerald-400 */
  border: 1px solid rgba(16, 185, 129, 0.35);
}

/* ── Recent activity feed (timestamped events) ─────────────────── */

.rade-feed {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.rade-feed-row {
  display: grid;
  grid-template-columns: 72px 1fr;
  align-items: baseline;
  column-gap: 0.75rem;
}

.rade-feed-time {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.7rem;
  color: #64748b;                     /* slate-500 */
}

.rade-feed-body {
  font-size: 0.8125rem;
  color: #cbd5e1;                     /* slate-300 */
}

.rade-feed-body-meta {
  margin-left: 0.5rem;
  color: #64748b;                     /* slate-500 */
  font-size: 0.75rem;
}

/* Small leading bullet before each feed row. */
.rade-feed-row::before {
  content: "";
  grid-column: 1;
  grid-row: 1;
  width: 6px;
  height: 6px;
  border-radius: 9999px;
  background-color: #8b5cf6;          /* violet-500 */
  align-self: center;
  transform: translateX(-16px);
  box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.18);
}

/* ── Quick actions footer strip ────────────────────────────────── */

.rade-quick-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  padding: 0.875rem 1.25rem;
  background-color: #0f172a;          /* slate-900 */
  border: 1px solid #1e293b;          /* slate-800 */
  border-radius: 0.75rem;
}

.rade-quick-actions-label {
  font-size: 0.9375rem;
  font-weight: 600;
  color: #e2e8f0;                     /* slate-200 */
}

.rade-quick-actions-group {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
```


---

## Appendix E — `callbacks/evaluation_cb.py` (Step 3, full file)

> **Usage.** **Replace the entire contents** of
> `src/ui/apps/rade_analytics/callbacks/evaluation_cb.py` with the
> script below.  It contains every callback from the E.0 skeleton plus
> the new `_register_metadata_hydration` pathway that populates the
> filter-bar dropdowns + date-picker bounds from
> `RadeBackend.evaluation_metadata`.
>
> Prerequisites: Steps 1 (`data/filters.py`) and 2 (`data/backend.py`)
> must already be applied — this script imports from both.
>
> Strip this appendix once the file is smoke-tested.

```python
"""Evaluation page callbacks.

Registers every interactive binding for the ``/evaluation/*`` routes:

1. **Sub-tab routing** — keeps URL, ``dmc.Tabs.value`` and the content
   slot in lock-step.  URL is the source of truth.
2. **Filter-bar collapse** — clicking the "Filters" button toggles the
   ``dmc.Collapse`` opened flag and persists the preference in session.
3. **Filter dropdowns → session** — whenever any MultiSelect or the
   DatePickerInput range changes, the full :class:`EvaluationFilters`
   payload is rebuilt and written back to session.  Chips + "n active"
   label are re-rendered from the session in the same callback so
   there's no race where UI and state disagree.
4. **Chip close** — pattern-matching callback that clears a single
   dimension when the user hits the ``×`` on its chip.
5. **Reset / Clear-all** — both empty every dimension at once.
6. **Hydration (values)** — restore filter-bar values from session
   whenever the user transitions *into* Evaluation from another page.
7. **Hydration (metadata)** — populate MultiSelect ``data`` and the
   DatePickerInput ``minDate`` / ``maxDate`` props from
   :meth:`RadeBackend.evaluation_metadata` on the same transition.

The callbacks read the live :class:`Session` from ``session-store`` via
``State`` and write back via ``Output`` so no module-level globals hold
mutable state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from dash import ALL, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from ..components.evaluation_filter_bar import (
    EVAL_FILTER_IDS,
    render_filter_chips,
)
from ..data.session import EvaluationFilters, Session
from ..layouts.evaluation.shell import (
    EVALUATION_IDS,
    active_subtab_from_path,
    build_subtab_content,
    path_for_subtab,
)
from ..layouts.shell import SHELL_IDS


if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


# ─────────────────────────────────────────────────────────────────────
# Styling constants — kept module-private so callbacks read by intent
# rather than inlining class strings / inline styles.
# ─────────────────────────────────────────────────────────────────────

_CLEAR_ALL_VISIBLE = {}                     # inherit default (inline)
_CLEAR_ALL_HIDDEN = {"display": "none"}


def _count_label(n: int) -> str:
    """Render the toggle button's "n active" sidecar label."""
    if n <= 0:
        return ""
    return f"{n} active"


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Evaluation callback to ``app``.

    ``backend`` is captured by closure in the data-hydration callbacks
    (metadata in this module; per-sub-tab fetches in E.1+).  Router and
    filter state machinery stays backend-agnostic.
    """
    _register_routing(app)
    _register_filter_bar(app)
    _register_hydration(app)
    _register_metadata_hydration(app, backend)


# ─────────────────────────────────────────────────────────────────────
# 1. Routing — URL ↔ dmc.Tabs ↔ content slot
# ─────────────────────────────────────────────────────────────────────


def _register_routing(app: "Dash") -> None:
    """URL → tab / content + tab click → URL."""

    # ── URL → tab value + content slot ───────────────────────────
    @app.callback(
        Output(EVALUATION_IDS["tabs"],    "value"),
        Output(EVALUATION_IDS["content"], "children"),
        Input(SHELL_IDS["url"], "pathname"),
    )
    def _sync_from_url(pathname: Optional[str]) -> Tuple[str, Any]:
        # Don't trigger on routes that aren't Evaluation — the outer
        # router is rebuilding the whole page_content anyway and the
        # Evaluation tree may not even exist in the DOM yet.
        if not pathname or not pathname.startswith("/evaluation"):
            raise PreventUpdate
        slug = active_subtab_from_path(pathname)
        return slug, build_subtab_content(slug)

    # ── Tab click → URL push ─────────────────────────────────────
    @app.callback(
        Output(SHELL_IDS["url"], "pathname", allow_duplicate=True),
        Input(EVALUATION_IDS["tabs"], "value"),
        State(SHELL_IDS["url"], "pathname"),
        prevent_initial_call=True,
    )
    def _push_url_from_tab(
        selected: Optional[str],
        current_pathname: Optional[str],
    ) -> str:
        if not selected:
            raise PreventUpdate
        new_path = path_for_subtab(selected)
        # Avoid an infinite loop with ``_sync_from_url`` — if the URL
        # is already where we'd push to, don't bounce.
        if current_pathname == new_path:
            raise PreventUpdate
        return new_path


# ─────────────────────────────────────────────────────────────────────
# 2. Filter bar — collapse, dropdowns, chips, clear / reset
# ─────────────────────────────────────────────────────────────────────


def _register_filter_bar(app: "Dash") -> None:
    """Wire the collapsible filter bar to ``session.evaluation.filters``."""

    # ── Toggle: Filters button → Collapse.opened + session preference ──
    @app.callback(
        Output(EVAL_FILTER_IDS["collapse"],   "opened"),
        Output(SHELL_IDS["session_store"],    "data", allow_duplicate=True),
        Input(EVAL_FILTER_IDS["toggle_btn"],  "n_clicks"),
        State(EVAL_FILTER_IDS["collapse"],    "opened"),
        State(SHELL_IDS["session_store"],     "data"),
        prevent_initial_call=True,
    )
    def _toggle_drawer(
        n_clicks: Optional[int],
        currently_open: Optional[bool],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate
        new_open = not bool(currently_open)

        session = Session.from_store(session_data)
        session.evaluation.filter_bar_open = new_open
        return new_open, session.to_store()

    # ── Dropdowns / date range → session + chips + label ─────────
    #
    # One unified callback so chip rendering can't race dropdown
    # writes.  Every Input below fires the same function; ctx tells us
    # which one triggered but we rebuild the whole EvaluationFilters
    # dataclass anyway so it rarely matters.
    @app.callback(
        Output(SHELL_IDS["session_store"],       "data",      allow_duplicate=True),
        Output(EVAL_FILTER_IDS["chips"],         "children"),
        Output(EVAL_FILTER_IDS["toggle_label"],  "children"),
        Output(EVAL_FILTER_IDS["clear_all"],     "style"),
        Input(EVAL_FILTER_IDS["asset_class"],    "value"),
        Input(EVAL_FILTER_IDS["currency"],       "value"),
        Input(EVAL_FILTER_IDS["desk"],           "value"),
        Input(EVAL_FILTER_IDS["product"],        "value"),
        Input(EVAL_FILTER_IDS["date_range"],     "value"),
        State(SHELL_IDS["session_store"],        "data"),
        prevent_initial_call=True,
    )
    def _sync_filters_to_session(
        asset_class: Optional[List[str]],
        currency:    Optional[List[str]],
        desk:        Optional[List[str]],
        product:     Optional[List[str]],
        date_range:  Optional[List[Optional[str]]],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Any], str, Dict[str, Any]]:
        session = Session.from_store(session_data)

        df, dt = _parse_date_range(date_range)
        filters = EvaluationFilters(
            asset_class=list(asset_class or []),
            currency=list(currency or []),
            desk=list(desk or []),
            product=list(product or []),
            date_from=df,
            date_to=dt,
        )
        session.evaluation.filters = filters

        chips = render_filter_chips(filters.to_dict())
        label = _count_label(filters.active_chip_count())
        clear_style = _CLEAR_ALL_VISIBLE if not filters.is_empty() else _CLEAR_ALL_HIDDEN

        return session.to_store(), chips, label, clear_style

    # ── Chip × (pattern-matching) — clear a single dimension ────
    @app.callback(
        Output(EVAL_FILTER_IDS["asset_class"], "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["currency"],    "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["desk"],        "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["product"],     "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["date_range"],  "value", allow_duplicate=True),
        Input({"type": "eval-filter-chip-close", "dimension": ALL}, "n_clicks"),
        State(EVAL_FILTER_IDS["asset_class"],  "value"),
        State(EVAL_FILTER_IDS["currency"],     "value"),
        State(EVAL_FILTER_IDS["desk"],         "value"),
        State(EVAL_FILTER_IDS["product"],      "value"),
        State(EVAL_FILTER_IDS["date_range"],   "value"),
        prevent_initial_call=True,
    )
    def _clear_single_filter(
        n_clicks_list: List[Optional[int]],
        asset_class:   Optional[List[str]],
        currency:      Optional[List[str]],
        desk:          Optional[List[str]],
        product:       Optional[List[str]],
        date_range:    Optional[List[Optional[str]]],
    ) -> Tuple[Any, Any, Any, Any, Any]:
        # pattern-matching callbacks fire on *every* matching id when
        # the layout first renders; bail until a real click lands.
        if not any(n_clicks_list or []):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        dim = triggered.get("dimension")

        # Default every Output to no_update so only the targeted
        # dropdown's value is rewritten — the unified sync callback
        # above then picks up the change and refreshes everything.
        updates: Dict[str, Any] = {
            EVAL_FILTER_IDS["asset_class"]: no_update,
            EVAL_FILTER_IDS["currency"]:    no_update,
            EVAL_FILTER_IDS["desk"]:        no_update,
            EVAL_FILTER_IDS["product"]:     no_update,
            EVAL_FILTER_IDS["date_range"]:  no_update,
        }

        if dim == "asset_class":
            updates[EVAL_FILTER_IDS["asset_class"]] = []
        elif dim == "currency":
            updates[EVAL_FILTER_IDS["currency"]] = []
        elif dim == "desk":
            updates[EVAL_FILTER_IDS["desk"]] = []
        elif dim == "product":
            updates[EVAL_FILTER_IDS["product"]] = []
        elif dim == "date":
            updates[EVAL_FILTER_IDS["date_range"]] = [None, None]
        else:
            raise PreventUpdate

        # Silence "unused" warnings — the State values above are only
        # here to keep the callback idempotent if we ever switch to a
        # granular update path.
        del asset_class, currency, desk, product, date_range

        return (
            updates[EVAL_FILTER_IDS["asset_class"]],
            updates[EVAL_FILTER_IDS["currency"]],
            updates[EVAL_FILTER_IDS["desk"]],
            updates[EVAL_FILTER_IDS["product"]],
            updates[EVAL_FILTER_IDS["date_range"]],
        )

    # ── Clear all / Reset — both empty every dimension ──────────
    @app.callback(
        Output(EVAL_FILTER_IDS["asset_class"], "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["currency"],    "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["desk"],        "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["product"],     "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["date_range"],  "value", allow_duplicate=True),
        Input(EVAL_FILTER_IDS["clear_all"],    "n_clicks"),
        Input(EVAL_FILTER_IDS["reset_btn"],    "n_clicks"),
        prevent_initial_call=True,
    )
    def _clear_all_filters(
        clear_clicks: Optional[int],
        reset_clicks: Optional[int],
    ) -> Tuple[List[str], List[str], List[str], List[str], List[Any]]:
        if not (clear_clicks or reset_clicks):
            raise PreventUpdate
        return [], [], [], [], [None, None]


# ─────────────────────────────────────────────────────────────────────
# 3. Hydration — restore filter-bar UI from session on Evaluation entry
# ─────────────────────────────────────────────────────────────────────


def _register_hydration(app: "Dash") -> None:
    """Push session-held filter state back into the filter-bar DOM.

    Fires whenever the user transitions *into* ``/evaluation`` from
    somewhere else.  The smart router (see ``router.py``) guarantees
    the filter bar is freshly mounted at this exact point, so the
    MultiSelect / DatePickerInput defaults need to be re-synced with
    whatever the user had set before navigating away.

    Crucially, this callback detects within-page navigation (e.g.
    ``/evaluation/portfolio`` → ``/evaluation/cross-cluster``) via the
    ``top_level_store`` State and bails out — the filter bar tree is
    still live, so there's nothing to hydrate.  This prevents a
    visible flicker and also sidesteps the sync callback clobbering
    user edits with stale session data on every tab click.
    """

    @app.callback(
        Output(EVAL_FILTER_IDS["asset_class"], "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["currency"],    "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["desk"],        "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["product"],     "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["date_range"],  "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["collapse"],    "opened", allow_duplicate=True),
        Input(SHELL_IDS["url"],                "pathname"),
        State(SHELL_IDS["session_store"],      "data"),
        State(SHELL_IDS["top_level_store"],    "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _hydrate_filter_bar(
        pathname:          Optional[str],
        session_data:      Optional[Dict[str, Any]],
        prev_top_level:    Optional[str],
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        # Only hydrate when *entering* Evaluation from elsewhere — a
        # within-page sub-tab click shares the same mounted filter bar
        # so re-pushing the same values would just race the user.
        if not pathname or not pathname.startswith("/evaluation"):
            raise PreventUpdate
        if prev_top_level == "/evaluation":
            # Within-page nav; filter bar DOM already reflects session.
            raise PreventUpdate

        session = Session.from_store(session_data)
        f = session.evaluation.filters
        return (
            list(f.asset_class),
            list(f.currency),
            list(f.desk),
            list(f.product),
            [f.date_from, f.date_to] if (f.date_from or f.date_to) else [None, None],
            bool(session.evaluation.filter_bar_open),
        )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _parse_date_range(
    value: Optional[List[Optional[str]]],
) -> Tuple[Optional[str], Optional[str]]:
    """Normalise ``dmc.DatePickerInput(type='range').value`` into (from, to).

    DMC returns ``[None, None]`` on a fresh / cleared picker, ``[iso]``
    on a first-date-only partial selection and ``[iso_from, iso_to]``
    on a full range.  Normalise each of those into a tidy 2-tuple.
    """
    if not value or not isinstance(value, (list, tuple)):
        return None, None
    df = value[0] if len(value) > 0 else None
    dt = value[1] if len(value) > 1 else None
    return (df or None), (dt or None)


# ─────────────────────────────────────────────────────────────────────
# 4. Metadata hydration — populate filter-bar dropdowns from backend
# ─────────────────────────────────────────────────────────────────────


def _options(values: List[str]) -> List[Dict[str, str]]:
    """Format a list of raw strings as ``dmc.MultiSelect`` data items."""
    return [{"value": v, "label": v} for v in values]


def _register_metadata_hydration(app: "Dash", backend: "RadeBackend") -> None:
    """Populate filter-bar dropdowns from :meth:`RadeBackend.evaluation_metadata`.

    Fires on the same transition as :func:`_register_hydration` (entry
    into ``/evaluation`` from another top-level page).  The existing
    value-sync callback takes over afterwards — nothing here writes to
    ``MultiSelect.value``.

    Kept as a separate callback from ``_hydrate_filter_bar`` because the
    outputs target different props (``data`` vs ``value``) and mixing
    them forces ``allow_duplicate=True`` unnecessarily on every value
    Output in the module.
    """

    @app.callback(
        Output(EVAL_FILTER_IDS["asset_class"], "data"),
        Output(EVAL_FILTER_IDS["currency"],    "data"),
        Output(EVAL_FILTER_IDS["desk"],        "data"),
        Output(EVAL_FILTER_IDS["product"],     "data"),
        Output(EVAL_FILTER_IDS["date_range"],  "minDate"),
        Output(EVAL_FILTER_IDS["date_range"],  "maxDate"),
        Input(SHELL_IDS["url"],             "pathname"),
        State(SHELL_IDS["top_level_store"], "data"),
    )
    def _load_metadata(
        pathname:       Optional[str],
        prev_top_level: Optional[str],
    ) -> Tuple[
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        Any,
        Any,
    ]:
        if not pathname or not pathname.startswith("/evaluation"):
            raise PreventUpdate
        if prev_top_level == "/evaluation":
            # Within-page sub-tab nav — dropdown data is still hot.
            raise PreventUpdate

        result = backend.evaluation_metadata()
        if not result.ok:
            # Leave dropdowns empty + date picker open.  The filter bar
            # degrades gracefully — the user just can't pick yet.
            return [], [], [], [], no_update, no_update

        meta = result.data
        return (
            _options(meta.asset_class),
            _options(meta.currency),
            _options(meta.desk),
            _options(meta.product),
            meta.date_min or no_update,
            meta.date_max or no_update,
        )


__all__ = ["register"]
```

### Verify

```bash
python -m py_compile \
  src/ui/apps/rade_analytics/callbacks/evaluation_cb.py
```

If it compiles and your IDE shows no new warnings, Step 3 is done.
