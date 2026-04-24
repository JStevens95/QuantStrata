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

---

## Appendix F — Phase E.1 Portfolio live-wiring (copy/paste)

> **Usage.**  Twelve paste-blocks total: **9 new files** + **3 edits**
> to existing files.  Apply them top-to-bottom — every block is a full
> drop-in for one file so you can copy/paste without thinking about
> diffs.
>
> **Why these files?**  Phase E.1 wires the Evaluation → Portfolio
> sub-tab end-to-end:
>
> * a new `figures/` package holds every chart builder (violin,
>   scatter, PnL, error-over-time);
> * the Portfolio layout gets replaced with the real 5-row structure
>   (KPI strip · PnL · error-over-time · error-analysis divider ·
>   faceted charts · leaderboard);
> * a single callback module drives render, group-by, and click-to-
>   focus behaviour;
> * a shared `_mock_backend.py` + per-page preview script provide a
>   smoke test with no running backend.
>
> Strip this appendix once the Portfolio tab renders live data (or
> mock data via the preview script) end-to-end.

### File table

| # | Action      | Path                                                                           | Purpose |
|---|-------------|--------------------------------------------------------------------------------|---------|
| 1 | **EDIT**    | `src/ui/apps/rade_analytics/data/session.py`                                   | Bump schema to v3 + add `portfolio_scatter_focus` |
| 2 | **CREATE**  | `src/ui/apps/rade_analytics/figures/_theme.py`                                 | Shared Plotly layout, palette, `rgba()` helper |
| 3 | **CREATE**  | `src/ui/apps/rade_analytics/figures/distributions.py`                          | Residual violin (aggregate + grouped) |
| 4 | **CREATE**  | `src/ui/apps/rade_analytics/figures/scatter.py`                                | Predicted-vs-actual scatter (aggregate / grouped / focus) |
| 5 | **CREATE**  | `src/ui/apps/rade_analytics/figures/timeseries.py`                             | Portfolio PnL + rolling-error band |
| 6 | **CREATE**  | `src/ui/apps/rade_analytics/figures/__init__.py`                               | Package init — re-exports every builder |
| 7 | **REPLACE** | `src/ui/apps/rade_analytics/layouts/evaluation/portfolio.py`                   | Full 5-row Portfolio layout |
| 8 | **CREATE**  | `src/ui/apps/rade_analytics/callbacks/portfolio_cb.py`                         | Render + group-by + click-to-focus callbacks |
| 9 | **EDIT**    | `src/ui/apps/rade_analytics/callbacks/__init__.py`                             | Register `portfolio_cb` alongside the others |
| 10 | **CREATE** | `examples/rade_analytics/_mock_backend.py`                                     | Shared mock backend (no API required) |
| 11 | **REPLACE**| `examples/rade_analytics/05_overview_preview_live.py`                          | Re-aim at the shared mock backend |
| 12 | **CREATE** | `examples/rade_analytics/06_portfolio_preview_live.py`                         | Portfolio smoke test on port 8053 |

### Smoke testing without a running backend

After every paste-block is in place:

```bash
python examples/rade_analytics/06_portfolio_preview_live.py
```

Open <http://localhost:8053/evaluation/portfolio>:

* Rows 1-3 should populate with KPI numbers, a purple PnL line + dashed
  actual, and a red rolling-error band.
* Pick a break-down dimension (Desk, Product, Currency, Asset class,
  Cluster) — the violin, scatter, and leaderboard should redraw with
  per-group colours.
* Click a point in the grouped scatter → the scatter filters to that
  group and a "Focused: X [× Show all]" chip appears in the card
  header.  Double-click the plot (or click the chip's ×) to clear.
* Toggle the topbar split control — every slot, including the
  leaderboard, must redraw.

The previous overview preview still runs via
`python examples/rade_analytics/05_overview_preview_live.py` on
port 8052 — it now shares the same `MockRadeBackend` instance shape so
numbers are consistent across the two pages.

---

### Step 1 · EDIT `src/ui/apps/rade_analytics/data/session.py`

> **Action.**  Replace the entire file contents with the block below.
> Schema version bumps from 2 → 3; any old session payloads in the
> browser's `sessionStorage` get dropped defensively by `from_store`.

```python
"""Typed per-user session state for the Rade Analytics UI.

Persisted to a ``dcc.Store(storage_type="session")`` as plain JSON.  The
contract here is:

* callbacks write a :class:`Session` instance, serialised via
  :meth:`Session.to_store`, into the store component;
* callbacks read the store via :meth:`Session.from_store`, which is
  defensive against partial / missing / stale payloads so the UI never
  crashes on a schema bump.

Adding a new field?  Bump :data:`SESSION_SCHEMA_VERSION` and leave the
field optional on :class:`Session`.  :meth:`from_store` will drop old
payloads whose version doesn't match.

Schema versions
---------------
* ``1`` — Phase A baseline (active_version, split, cluster_id, theme).
* ``2`` — Phase E.0 adds :class:`EvaluationState` under ``evaluation``.
* ``3`` — Phase E.1 adds ``portfolio_scatter_focus`` to
  :class:`EvaluationState` for the click-to-focus scatter behaviour.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

SESSION_SCHEMA_VERSION = 3


# ─────────────────────────────────────────────────────────────────────
# Evaluation sub-state (Phase E.0)
# ─────────────────────────────────────────────────────────────────────

EVALUATION_SUBTABS = ("portfolio", "cross-cluster", "trade-graph", "cluster")
DEFAULT_EVALUATION_SUBTAB: str = "portfolio"

EVALUATION_PORTFOLIO_GROUP_BY = (
    "desk", "product", "currency", "asset_class", "cluster",
)


@dataclass
class EvaluationFilters:
    """The global WHERE clause applied across every Evaluation sub-tab."""

    asset_class: List[str] = field(default_factory=list)
    currency:    List[str] = field(default_factory=list)
    desk:        List[str] = field(default_factory=list)
    product:     List[str] = field(default_factory=list)
    date_from:   Optional[str] = None
    date_to:     Optional[str] = None

    def active_chip_count(self) -> int:
        return sum(
            bool(x) for x in (
                self.asset_class,
                self.currency,
                self.desk,
                self.product,
                self.date_from or self.date_to,
            )
        )

    def is_empty(self) -> bool:
        return self.active_chip_count() == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EvaluationFilters":
        if not isinstance(data, dict):
            return cls()
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class EvaluationState:
    """Everything the Evaluation page needs between sub-tab switches."""

    filters:                   EvaluationFilters = field(default_factory=EvaluationFilters)
    filter_bar_open:           bool = False
    active_subtab:             str = DEFAULT_EVALUATION_SUBTAB
    portfolio_group_by:        Optional[str] = None
    portfolio_scatter_focus:   Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filters":                 self.filters.to_dict(),
            "filter_bar_open":         self.filter_bar_open,
            "active_subtab":           self.active_subtab,
            "portfolio_group_by":      self.portfolio_group_by,
            "portfolio_scatter_focus": self.portfolio_scatter_focus,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EvaluationState":
        if not isinstance(data, dict):
            return cls()

        subtab = data.get("active_subtab", DEFAULT_EVALUATION_SUBTAB)
        if subtab not in EVALUATION_SUBTABS:
            subtab = DEFAULT_EVALUATION_SUBTAB

        group_by = data.get("portfolio_group_by")
        if group_by is not None and group_by not in EVALUATION_PORTFOLIO_GROUP_BY:
            group_by = None

        scatter_focus = data.get("portfolio_scatter_focus")
        if group_by is None:
            scatter_focus = None
        if scatter_focus is not None and not isinstance(scatter_focus, str):
            scatter_focus = None

        return cls(
            filters=EvaluationFilters.from_dict(data.get("filters")),
            filter_bar_open=bool(data.get("filter_bar_open", False)),
            active_subtab=subtab,
            portfolio_group_by=group_by,
            portfolio_scatter_focus=scatter_focus,
        )


# ─────────────────────────────────────────────────────────────────────
# Top-level session
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Session:
    """User session state.  Every field must be JSON-serialisable."""

    active_version: Optional[str] = None

    split: Literal["train", "val", "test"] = "test"
    cluster_id: Optional[str] = None

    theme: Literal["dark", "light"] = "dark"

    evaluation: EvaluationState = field(default_factory=EvaluationState)

    schema_version: int = field(default=SESSION_SCHEMA_VERSION, repr=False)

    def to_store(self) -> Dict[str, Any]:
        return {
            "active_version": self.active_version,
            "split":          self.split,
            "cluster_id":     self.cluster_id,
            "theme":          self.theme,
            "evaluation":     self.evaluation.to_dict(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_store(cls, data: Optional[Dict[str, Any]]) -> "Session":
        if not data or not isinstance(data, dict):
            return cls()
        if data.get("schema_version") != SESSION_SCHEMA_VERSION:
            return cls()
        split = data.get("split", "test")
        if split not in ("train", "val", "test"):
            split = "test"
        theme = data.get("theme", "dark")
        if theme not in ("dark", "light"):
            theme = "dark"
        return cls(
            active_version=data.get("active_version"),
            split=split,                                      # type: ignore[arg-type]
            cluster_id=data.get("cluster_id"),
            theme=theme,                                      # type: ignore[arg-type]
            evaluation=EvaluationState.from_dict(data.get("evaluation")),
        )

    def with_version(self, version: str) -> "Session":
        payload = self.to_store()
        payload["active_version"] = version
        return Session.from_store(payload)

    def with_split(self, split: str) -> "Session":
        payload = self.to_store()
        payload["split"] = split
        return Session.from_store(payload)

    def with_cluster(self, cluster_id: Optional[str]) -> "Session":
        payload = self.to_store()
        payload["cluster_id"] = cluster_id
        return Session.from_store(payload)

    def with_evaluation(self, evaluation: EvaluationState) -> "Session":
        payload = self.to_store()
        payload["evaluation"] = evaluation.to_dict()
        return Session.from_store(payload)


__all__ = [
    "DEFAULT_EVALUATION_SUBTAB",
    "EVALUATION_PORTFOLIO_GROUP_BY",
    "EVALUATION_SUBTABS",
    "EvaluationFilters",
    "EvaluationState",
    "SESSION_SCHEMA_VERSION",
    "Session",
]
```

---

### Step 2 · CREATE `src/ui/apps/rade_analytics/figures/_theme.py`

> **Action.**  Create a new `figures/` directory under
> `src/ui/apps/rade_analytics/` and drop this file inside.

```python
"""Shared plotly layout defaults for every Rade figure."""
from __future__ import annotations

from typing import Any, Dict, Optional

import plotly.graph_objects as go


_FONT_FAMILY = "Inter, system-ui, sans-serif"
_FONT_COLOR = "#cbd5e1"
_TICK_COLOR = "#64748b"
_GRID_COLOR = "rgba(30, 41, 59, 0.6)"


CATEGORY_PALETTE: tuple[str, ...] = (
    "#8b5cf6",  # violet (primary)
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#f43f5e",  # rose
    "#38bdf8",  # sky
    "#06b6d4",  # cyan
    "#d946ef",  # fuchsia
    "#84cc16",  # lime
)


def color_for_index(i: int) -> str:
    return CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]


def rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex string to rgba notation.

    Plotly's violin / scatter validators refuse 8-digit hex alphas, so
    every fillcolor in this package goes through here.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"rgba() expects a 6-digit hex colour, got {hex_color!r}")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r}, {g}, {b}, {a:.3f})"


def rade_layout(
    *,
    show_legend: bool = False,
    hovermode: str = "closest",
    margin: Optional[Dict[str, int]] = None,
    xaxis: Optional[Dict[str, Any]] = None,
    yaxis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_xaxis = {
        "showgrid":       False,
        "zeroline":       False,
        "showticklabels": True,
        "tickfont":       {"color": _TICK_COLOR},
    }
    base_yaxis = {
        "gridcolor": _GRID_COLOR,
        "zeroline":  False,
        "tickfont":  {"color": _TICK_COLOR},
    }
    if xaxis:
        base_xaxis.update(xaxis)
    if yaxis:
        base_yaxis.update(yaxis)

    return {
        "template":     "plotly_dark",
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "margin":       margin or {"l": 40, "r": 16, "t": 8, "b": 36},
        "font":         {"family": _FONT_FAMILY, "color": _FONT_COLOR, "size": 11},
        "xaxis":        base_xaxis,
        "yaxis":        base_yaxis,
        "hovermode":    hovermode,
        "showlegend":   show_legend,
        "legend": {
            "orientation": "h",
            "y": 1.1, "x": 1, "xanchor": "right",
            "bgcolor": "rgba(0, 0, 0, 0)",
            "font": {"color": "#94a3b8", "size": 11},
        },
    }


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **rade_layout(
            margin={"l": 32, "r": 16, "t": 8, "b": 32},
            xaxis={"visible": False},
            yaxis={"visible": False},
        ),
        annotations=[
            {
                "text":      message,
                "showarrow": False,
                "font":      {"color": "#64748b", "size": 13},
                "xref":      "paper",
                "yref":      "paper",
                "x":         0.5,
                "y":         0.5,
            }
        ],
    )
    return fig


__all__ = [
    "CATEGORY_PALETTE",
    "color_for_index",
    "empty_figure",
    "rade_layout",
    "rgba",
]
```

---

### Step 3 · CREATE `src/ui/apps/rade_analytics/figures/distributions.py`

> **Action.**  Create the file.  Uses `rgba()` from `_theme` for every
> fillcolor — required for Plotly's validator compatibility.

```python
"""Residual distribution figures — violin plots (aggregate + grouped)."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from ._theme import color_for_index, empty_figure, rade_layout, rgba


def residual_violin(
    residuals:     Sequence[float],
    *,
    group_values:  Optional[Sequence[str]] = None,
    group_order:   Optional[Iterable[str]] = None,
    group_label:   Optional[str] = None,
    y_axis_title:  str = "Residual (pred − actual)",
) -> go.Figure:
    """Residual distribution violin.  Aggregate when ``group_values`` is ``None``."""
    arr = np.asarray(list(residuals), dtype=float)
    if arr.size == 0:
        return empty_figure("No residuals to plot.")

    grouped = group_values is not None and len(group_values) == arr.size

    if not grouped:
        return _aggregate_violin(arr, y_axis_title=y_axis_title)

    groups = _resolve_group_order(group_values, group_order)
    if not groups:
        return _aggregate_violin(arr, y_axis_title=y_axis_title)

    group_arr = np.asarray(group_values, dtype=object)
    fig = go.Figure()
    for idx, group in enumerate(groups):
        mask = group_arr == group
        values = arr[mask]
        if values.size == 0:
            continue
        fig.add_trace(
            go.Violin(
                y=values,
                name=str(group),
                x=[str(group)] * values.size,
                line_color=color_for_index(idx),
                fillcolor=rgba(color_for_index(idx), 0.2),
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                hoveron="violins",
                hovertemplate=(
                    f"<b>{group}</b><br>"
                    "median: %{median:.4f}<br>"
                    "Q1: %{q1:.4f} / Q3: %{q3:.4f}<br>"
                    "min: %{lowerfence:.4f} / max: %{upperfence:.4f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        **rade_layout(
            show_legend=False,
            xaxis={
                "title": {
                    "text": group_label or "",
                    "font": {"color": "#94a3b8", "size": 11},
                },
                "showgrid": False,
            },
            yaxis={"title": {"text": y_axis_title, "font": {"color": "#94a3b8"}}},
        ),
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.4)",
        line_width=1,
    )
    return fig


def _aggregate_violin(
    residuals: np.ndarray,
    *,
    y_axis_title: str,
) -> go.Figure:
    mu    = float(residuals.mean())
    sigma = float(residuals.std())
    if sigma > 0:
        pct_1s = float(np.mean(np.abs(residuals - mu) <= sigma) * 100)
        pct_2s = float(np.mean(np.abs(residuals - mu) <= 2 * sigma) * 100)
        skew, kurt = _skew_kurt(residuals)
    else:
        pct_1s = pct_2s = 100.0
        skew = kurt = 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=residuals,
            x=[""] * residuals.size,
            name="Residuals",
            line_color=color_for_index(0),
            fillcolor=rgba(color_for_index(0), 0.2),
            box_visible=True,
            meanline_visible=True,
            points="outliers",
            hoveron="violins",
            showlegend=False,
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.4)",
        line_width=1,
    )

    annotation_text = (
        f"μ={mu:.4f}  σ={sigma:.4f}<br>"
        f"skew={skew:.2f}  kurt={kurt:.2f}<br>"
        f"±1σ: {pct_1s:.1f}%  ±2σ: {pct_2s:.1f}%  "
        f"n={residuals.size:,}"
    )

    fig.update_layout(
        **rade_layout(
            xaxis={"visible": False},
            yaxis={"title": {"text": y_axis_title, "font": {"color": "#94a3b8"}}},
        ),
        annotations=[
            {
                "text":      annotation_text,
                "showarrow": False,
                "xref":      "paper",
                "yref":      "paper",
                "x":         0.98,
                "y":         0.98,
                "xanchor":   "right",
                "yanchor":   "top",
                "bgcolor":   "rgba(15, 23, 42, 0.85)",
                "bordercolor": "rgba(148, 163, 184, 0.3)",
                "borderwidth": 1,
                "borderpad":   6,
                "font": {"size": 11, "color": "#cbd5e1", "family": "Inter, system-ui"},
                "align":     "right",
            }
        ],
    )
    return fig


def _skew_kurt(residuals: np.ndarray) -> tuple[float, float]:
    try:
        from scipy import stats as _stats  # noqa: WPS433 — lazy import by design
        return float(_stats.skew(residuals)), float(_stats.kurtosis(residuals))
    except Exception:  # noqa: BLE001
        mu = residuals.mean()
        sigma = residuals.std()
        if sigma <= 0:
            return 0.0, 0.0
        z = (residuals - mu) / sigma
        return float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def _resolve_group_order(
    group_values: Optional[Sequence[str]],
    group_order:  Optional[Iterable[str]],
) -> list[str]:
    if not group_values:
        return []
    uniq = sorted({str(g) for g in group_values if g is not None})
    if group_order is None:
        return uniq

    ordered: list[str] = []
    seen: set[str] = set()
    for g in group_order:
        key = str(g)
        if key in uniq and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in uniq:
        if key not in seen:
            ordered.append(key)
    return ordered


__all__ = ["residual_violin"]
```

---

### Step 4 · CREATE `src/ui/apps/rade_analytics/figures/scatter.py`

> **Action.**  Create the file.

```python
"""Predicted-vs-actual scatter figure (aggregate / grouped / focus)."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from ._theme import color_for_index, empty_figure, rade_layout


def pred_actual_scatter(
    predicted:     Sequence[float],
    actual:        Sequence[float],
    *,
    group_values:  Optional[Sequence[str]] = None,
    group_order:   Optional[Iterable[str]] = None,
    focus_group:   Optional[str] = None,
    hover_labels:  Optional[Sequence[str]] = None,
    x_axis_title:  str = "Predicted",
    y_axis_title:  str = "Actual",
) -> go.Figure:
    pred = np.asarray(list(predicted), dtype=float)
    act  = np.asarray(list(actual),    dtype=float)
    if pred.size == 0 or pred.shape != act.shape:
        return empty_figure("No prediction-actual pairs to plot.")

    all_vals = np.concatenate([pred, act])
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    span = (lo - pad, hi + pad)

    grouped = group_values is not None and len(group_values) == pred.size

    fig = go.Figure()

    if not grouped:
        _trace_aggregate(
            fig, pred, act,
            hover_labels=hover_labels,
            color=color_for_index(0),
        )
    else:
        groups = _resolve_group_order(group_values, group_order)
        group_arr = np.asarray(group_values, dtype=object)

        for idx, group in enumerate(groups):
            if focus_group is not None and group != focus_group:
                continue
            mask = group_arr == group
            if not mask.any():
                continue
            labels_here = (
                [hover_labels[i] for i in np.flatnonzero(mask)]
                if hover_labels is not None
                else None
            )
            _trace_one_group(
                fig,
                pred[mask], act[mask],
                group=str(group),
                color=color_for_index(idx),
                hover_labels=labels_here,
            )

    _add_identity_line(fig, span)

    fig.update_layout(
        **rade_layout(
            show_legend=grouped and focus_group is None,
            xaxis={
                "title": {"text": x_axis_title, "font": {"color": "#94a3b8"}},
                "range": list(span),
            },
            yaxis={
                "title": {"text": y_axis_title, "font": {"color": "#94a3b8"}},
                "range": list(span),
            },
        ),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _trace_aggregate(
    fig: go.Figure,
    pred: np.ndarray,
    act: np.ndarray,
    *,
    hover_labels: Optional[Sequence[str]],
    color: str,
) -> None:
    customdata = _build_customdata(group="", labels=hover_labels, n=pred.size)
    hovertemplate = (
        "pred: %{x:.4f}<br>actual: %{y:.4f}"
        "<br>%{customdata[1]}<extra></extra>"
        if hover_labels is not None else
        "pred: %{x:.4f}<br>actual: %{y:.4f}<extra></extra>"
    )
    fig.add_trace(
        go.Scattergl(
            x=pred, y=act,
            mode="markers",
            marker={
                "size":    6,
                "color":   color,
                "opacity": 0.7,
                "line":    {"width": 0},
            },
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="All",
            showlegend=False,
        )
    )


def _trace_one_group(
    fig: go.Figure,
    pred: np.ndarray,
    act: np.ndarray,
    *,
    group: str,
    color: str,
    hover_labels: Optional[Sequence[str]],
) -> None:
    customdata = _build_customdata(group=group, labels=hover_labels, n=pred.size)
    if hover_labels is not None:
        hovertemplate = (
            f"<b>{group}</b><br>"
            "pred: %{x:.4f}<br>actual: %{y:.4f}<br>"
            "%{customdata[1]}"
            "<extra>Click to focus</extra>"
        )
    else:
        hovertemplate = (
            f"<b>{group}</b><br>"
            "pred: %{x:.4f}<br>actual: %{y:.4f}"
            "<extra>Click to focus</extra>"
        )
    fig.add_trace(
        go.Scattergl(
            x=pred, y=act,
            mode="markers",
            marker={
                "size":    6,
                "color":   color,
                "opacity": 0.75,
                "line":    {"width": 0},
            },
            name=group,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )


def _add_identity_line(fig: go.Figure, span: tuple[float, float]) -> None:
    fig.add_trace(
        go.Scatter(
            x=list(span), y=list(span),
            mode="lines",
            line={"color": "rgba(148, 163, 184, 0.55)", "dash": "dash", "width": 1},
            hoverinfo="skip",
            showlegend=False,
            name="Identity",
        )
    )


def _build_customdata(
    *,
    group: str,
    labels: Optional[Sequence[str]],
    n: int,
) -> list[list[str]]:
    if labels is None:
        return [[group, ""] for _ in range(n)]
    return [[group, str(labels[i]) if labels[i] is not None else ""] for i in range(n)]


def _resolve_group_order(
    group_values: Optional[Sequence[str]],
    group_order:  Optional[Iterable[str]],
) -> list[str]:
    if not group_values:
        return []
    uniq = sorted({str(g) for g in group_values if g is not None})
    if group_order is None:
        return uniq

    ordered: list[str] = []
    seen: set[str] = set()
    for g in group_order:
        key = str(g)
        if key in uniq and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in uniq:
        if key not in seen:
            ordered.append(key)
    return ordered


__all__ = ["pred_actual_scatter"]
```

---

### Step 5 · CREATE `src/ui/apps/rade_analytics/figures/timeseries.py`

> **Action.**  Create the file.

```python
"""Portfolio-level time-series figures."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import color_for_index, empty_figure, rade_layout, rgba


def portfolio_pnl(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty or "predicted" not in df.columns or "actual" not in df.columns:
        return empty_figure("No portfolio data for the active filter set.")

    df_sorted = df.sort_values("scenario_idx") if "scenario_idx" in df.columns else df
    x_vals = (
        df_sorted["scenario_label"]
        if "scenario_label" in df_sorted.columns
        else list(range(len(df_sorted)))
    )

    pred_color = color_for_index(0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_sorted["predicted"],
            mode="lines",
            name="Predicted PnL",
            line={"color": pred_color, "width": 2.5},
            fill="tozeroy",
            fillcolor=rgba(pred_color, 0.18),
            hovertemplate="%{y:.4f}<extra>Predicted</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df_sorted["actual"],
            mode="lines",
            name="Actual PnL",
            line={"color": "#cbd5e1", "width": 1.5, "dash": "dash"},
            hovertemplate="%{y:.4f}<extra>Actual</extra>",
        )
    )
    fig.update_layout(
        **rade_layout(
            show_legend=True,
            hovermode="x unified",
            xaxis={"showticklabels": True},
            yaxis={"title": {"text": "PnL", "font": {"color": "#94a3b8"}}},
        ),
    )
    return fig


def error_over_time(
    df:       pd.DataFrame,
    *,
    window:   Optional[int] = None,
    band_std: float = 1.0,
) -> go.Figure:
    if df is None or df.empty:
        return empty_figure("No error-over-time data for the active filter set.")

    if "abs_error" in df.columns:
        abs_err = df["abs_error"].astype(float)
    elif {"predicted", "actual"}.issubset(df.columns):
        abs_err = (df["predicted"] - df["actual"]).abs().astype(float)
    else:
        return empty_figure("Error-over-time requires predicted + actual columns.")

    df_sorted = df.assign(_abs_error=abs_err)
    if "scenario_idx" in df_sorted.columns:
        df_sorted = df_sorted.sort_values("scenario_idx")
    x_vals = (
        df_sorted["scenario_label"]
        if "scenario_label" in df_sorted.columns
        else list(range(len(df_sorted)))
    )

    n = len(df_sorted)
    win = int(window) if window else max(3, n // 10)
    win = min(max(win, 3), n)

    rolling_mean = df_sorted["_abs_error"].rolling(win, min_periods=1).mean()
    rolling_std  = df_sorted["_abs_error"].rolling(win, min_periods=1).std().fillna(0.0)
    upper = rolling_mean + band_std * rolling_std
    lower = (rolling_mean - band_std * rolling_std).clip(lower=0)

    err_color = color_for_index(3)  # rose

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=upper,
            mode="lines",
            line={"color": "rgba(244, 63, 94, 0)", "width": 0},
            hoverinfo="skip",
            showlegend=False,
            name="upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=lower,
            mode="lines",
            line={"color": "rgba(244, 63, 94, 0)", "width": 0},
            fill="tonexty",
            fillcolor="rgba(244, 63, 94, 0.18)",
            hoverinfo="skip",
            showlegend=False,
            name="lower",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=df_sorted["_abs_error"],
            mode="lines",
            line={"color": "rgba(244, 63, 94, 0.35)", "width": 1},
            name="abs error",
            hovertemplate="%{y:.4f}<extra>abs error</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=rolling_mean,
            mode="lines",
            line={"color": err_color, "width": 2.5},
            name=f"rolling mean (w={win})",
            hovertemplate="%{y:.4f}<extra>rolling mean</extra>",
        )
    )

    fig.update_layout(
        **rade_layout(
            show_legend=True,
            hovermode="x unified",
            xaxis={"showticklabels": True},
            yaxis={"title": {"text": "Absolute error", "font": {"color": "#94a3b8"}}},
        ),
    )
    return fig


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


__all__ = ["error_over_time", "portfolio_pnl"]
```

---

### Step 6 · CREATE `src/ui/apps/rade_analytics/figures/__init__.py`

> **Action.**  Create the file — this makes `figures/` a proper Python
> package and exposes every builder from a single import.

```python
"""Figure builders for the Rade Analytics Dash UI."""
from __future__ import annotations

from ._theme import (
    CATEGORY_PALETTE,
    color_for_index,
    empty_figure,
    rade_layout,
    rgba,
)
from .distributions import residual_violin
from .scatter import pred_actual_scatter
from .timeseries import error_over_time, portfolio_pnl

__all__ = [
    "CATEGORY_PALETTE",
    "color_for_index",
    "empty_figure",
    "error_over_time",
    "portfolio_pnl",
    "pred_actual_scatter",
    "rade_layout",
    "residual_violin",
    "rgba",
]
```

---

### Step 7 · REPLACE `src/ui/apps/rade_analytics/layouts/evaluation/portfolio.py`

> **Action.**  Replace the entire file.  The old stub is gone; this is
> the real 5-row Portfolio layout.

```python
"""Evaluation → Portfolio sub-tab layout."""
from __future__ import annotations

from typing import Any, Dict, List

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify

from ...components.ag_grid_table import AgGridTable
from ...components.chart_container import ChartContainer
from ...components.kpi_card import KpiCard
from ...data.session import EVALUATION_PORTFOLIO_GROUP_BY


PORTFOLIO_IDS: Dict[str, str] = {
    "root":                 "eval-portfolio-root",

    "kpi_mae_card":         "eval-portfolio-kpi-mae-card",
    "kpi_mae_value":        "eval-portfolio-kpi-mae-value",
    "kpi_rmse_card":        "eval-portfolio-kpi-rmse-card",
    "kpi_rmse_value":       "eval-portfolio-kpi-rmse-value",
    "kpi_hit_rate_card":    "eval-portfolio-kpi-hit-rate-card",
    "kpi_hit_rate_value":   "eval-portfolio-kpi-hit-rate-value",
    "kpi_coverage_card":    "eval-portfolio-kpi-coverage-card",
    "kpi_coverage_value":   "eval-portfolio-kpi-coverage-value",

    "pnl_chart":            "eval-portfolio-pnl-chart",
    "error_ts_chart":       "eval-portfolio-error-ts-chart",

    "analysis_divider":     "eval-portfolio-analysis-divider",
    "groupby_select":       "eval-portfolio-groupby-select",
    "groupby_clear_btn":    "eval-portfolio-groupby-clear-btn",
    "groupby_count_label":  "eval-portfolio-groupby-count-label",

    "residual_violin":      "eval-portfolio-residual-violin",
    "pred_actual_scatter":  "eval-portfolio-pred-actual-scatter",

    "focus_chip_container": "eval-portfolio-focus-chip-container",
    "focus_chip_label":     "eval-portfolio-focus-chip-label",
    "focus_chip_clear_btn": "eval-portfolio-focus-chip-clear-btn",

    "leaderboard_card":      "eval-portfolio-leaderboard-card",
    "leaderboard_grid":      "eval-portfolio-leaderboard-grid",
    "leaderboard_grid_wrap": "eval-portfolio-leaderboard-grid-wrap",
    "leaderboard_empty":     "eval-portfolio-leaderboard-empty",
    "leaderboard_header":    "eval-portfolio-leaderboard-header",
}


_GROUP_BY_LABELS: Dict[str, str] = {
    "desk":        "Desk",
    "product":     "Product",
    "currency":    "Currency",
    "asset_class": "Asset class",
    "cluster":     "Cluster",
}

assert set(_GROUP_BY_LABELS.keys()) == set(EVALUATION_PORTFOLIO_GROUP_BY), (
    "Portfolio group-by labels out of sync with session whitelist"
)


def _groupby_options() -> List[Dict[str, str]]:
    return [
        {"value": key, "label": _GROUP_BY_LABELS[key]}
        for key in EVALUATION_PORTFOLIO_GROUP_BY
    ]


def build_portfolio() -> html.Div:
    return html.Div(
        id=PORTFOLIO_IDS["root"],
        className="rade-evaluation-subtab flex flex-col gap-4",
        children=[
            _row_kpis(),
            _row_pnl_chart(),
            _row_error_timeseries(),
            _row_analysis_divider(),
            _row_faceted_charts(),
            _row_leaderboard(),
        ],
    )


def _row_kpis() -> html.Div:
    return html.Div(
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3",
        children=[
            KpiCard(
                label="MAE",
                value="—",
                card_id=PORTFOLIO_IDS["kpi_mae_card"],
                value_id=PORTFOLIO_IDS["kpi_mae_value"],
                icon="tabler:arrow-narrow-down",
            ),
            KpiCard(
                label="RMSE",
                value="—",
                card_id=PORTFOLIO_IDS["kpi_rmse_card"],
                value_id=PORTFOLIO_IDS["kpi_rmse_value"],
                icon="tabler:square-root",
            ),
            KpiCard(
                label="Hit rate",
                value="—",
                card_id=PORTFOLIO_IDS["kpi_hit_rate_card"],
                value_id=PORTFOLIO_IDS["kpi_hit_rate_value"],
                icon="tabler:target",
            ),
            KpiCard(
                label="Coverage",
                value="—",
                card_id=PORTFOLIO_IDS["kpi_coverage_card"],
                value_id=PORTFOLIO_IDS["kpi_coverage_value"],
                icon="tabler:chart-bar",
            ),
        ],
    )


def _row_pnl_chart() -> html.Div:
    return ChartContainer(
        title="Portfolio PnL",
        subtitle="Predicted vs actual across the active split",
        graph_id=PORTFOLIO_IDS["pnl_chart"],
        height=300,
    )


def _row_error_timeseries() -> html.Div:
    return ChartContainer(
        title="Error over time",
        subtitle="Rolling absolute error with ±1σ band",
        graph_id=PORTFOLIO_IDS["error_ts_chart"],
        height=260,
    )


def _row_analysis_divider() -> html.Div:
    return html.Div(
        id=PORTFOLIO_IDS["analysis_divider"],
        className="rade-section-divider flex items-center gap-3",
        children=[
            html.Div(
                className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400",
                children=[
                    DashIconify(icon="tabler:chart-dots-2", width=14),
                    html.Span("Error analysis"),
                ],
            ),
            html.Div(className="flex-1 h-px bg-slate-800"),
            html.Div(
                className="flex items-center gap-2",
                children=[
                    html.Span(
                        "Break down by",
                        className="text-xs text-slate-400",
                    ),
                    dmc.Select(
                        id=PORTFOLIO_IDS["groupby_select"],
                        placeholder="None",
                        data=_groupby_options(),
                        value=None,
                        clearable=True,
                        searchable=False,
                        size="xs",
                        radius="sm",
                        className="rade-portfolio-groupby-select",
                        style={"minWidth": "160px"},
                    ),
                    html.Span(
                        id=PORTFOLIO_IDS["groupby_count_label"],
                        className="text-xs text-slate-500",
                        children="",
                    ),
                    dmc.Button(
                        id=PORTFOLIO_IDS["groupby_clear_btn"],
                        children="Clear",
                        leftSection=DashIconify(icon="tabler:x", width=12),
                        size="xs",
                        variant="subtle",
                        color="gray",
                        style={"display": "none"},
                    ),
                ],
            ),
        ],
    )


def _row_faceted_charts() -> html.Div:
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-2 gap-3",
        children=[
            ChartContainer(
                title="Residual distribution",
                subtitle="Pred − actual",
                graph_id=PORTFOLIO_IDS["residual_violin"],
                height=360,
            ),
            ChartContainer(
                title="Predicted vs actual",
                subtitle="Dashed line is the identity",
                graph_id=PORTFOLIO_IDS["pred_actual_scatter"],
                height=360,
                actions=[_focus_chip()],
                config={"doubleClick": "reset"},
            ),
        ],
    )


def _focus_chip() -> html.Div:
    return html.Div(
        id=PORTFOLIO_IDS["focus_chip_container"],
        className="rade-focus-chip flex items-center gap-1",
        style={"display": "none"},
        children=[
            DashIconify(icon="tabler:focus-2", width=12, className="text-violet-400"),
            html.Span(
                "Focused: —",
                id=PORTFOLIO_IDS["focus_chip_label"],
                className="text-xs text-slate-300",
            ),
            html.Button(
                "× Show all",
                id=PORTFOLIO_IDS["focus_chip_clear_btn"],
                className="rade-focus-chip-close",
                **{"aria-label": "Clear scatter focus"},
            ),
        ],
    )


def _row_leaderboard() -> html.Div:
    header = html.Div(
        id=PORTFOLIO_IDS["leaderboard_header"],
        className="flex items-center justify-between",
        children=[
            html.Div(
                className="flex flex-col",
                children=[
                    html.Div("Leaderboard", className="text-sm font-semibold text-slate-200"),
                    html.Div(
                        "Pick a break-down dimension to compare contributors.",
                        className="text-xs text-slate-500",
                    ),
                ],
            ),
        ],
    )

    empty_state = html.Div(
        id=PORTFOLIO_IDS["leaderboard_empty"],
        className="rade-list-empty flex flex-col items-center justify-center gap-2 py-8",
        children=[
            DashIconify(icon="tabler:table-off", width=22, className="text-slate-600"),
            html.Div(
                "Pick a break-down dimension above to compare contributors "
                "by desk, product, currency, asset class or cluster.",
                className="text-xs text-slate-500 text-center max-w-sm",
            ),
        ],
    )

    grid = AgGridTable(
        grid_id=PORTFOLIO_IDS["leaderboard_grid"],
        column_defs=_initial_column_defs(),
        row_data=[],
        height=320,
        className="rade-portfolio-leaderboard",
    )

    grid_wrapper = html.Div(
        id=PORTFOLIO_IDS["leaderboard_grid_wrap"],
        className="rade-portfolio-leaderboard-grid-wrap",
        style={"display": "none"},
        children=grid,
    )

    return html.Div(
        id=PORTFOLIO_IDS["leaderboard_card"],
        className="rade-card flex flex-col gap-3",
        children=[header, empty_state, grid_wrapper],
    )


def _initial_column_defs() -> List[Dict[str, Any]]:
    return [
        {"field": "group_label", "headerName": "Break-down", "flex": 2, "minWidth": 140},
        {"field": "mae",         "headerName": "MAE",        "flex": 1, "type": "numericColumn"},
        {"field": "rmse",        "headerName": "RMSE",       "flex": 1, "type": "numericColumn"},
        {"field": "hit_rate",    "headerName": "Hit %",      "flex": 1, "type": "numericColumn"},
        {"field": "contribution", "headerName": "Contribution", "flex": 1, "type": "numericColumn"},
        {"field": "n_clusters",  "headerName": "Clusters",   "flex": 1, "type": "numericColumn"},
    ]


__all__ = ["PORTFOLIO_IDS", "build_portfolio"]
```

---

### Step 8 · CREATE `src/ui/apps/rade_analytics/callbacks/portfolio_cb.py`

> **Action.**  Create the file.  This is the biggest single paste in
> the appendix — six callbacks + helpers.

```python
"""Evaluation → Portfolio sub-tab callbacks (Phase E.1)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from ..data.session import EvaluationFilters, Session
from ..figures import (
    empty_figure,
    error_over_time,
    pred_actual_scatter,
    portfolio_pnl,
    residual_violin,
)
from ..layouts.evaluation.portfolio import PORTFOLIO_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import BackendResult, RadeBackend


logger = logging.getLogger(__name__)


_PORTFOLIO_PATH = "/evaluation/portfolio"
_EVALUATION_PREFIX = "/evaluation"
_PLACEHOLDER = "—"

_GROUP_BY_COLUMN: Dict[str, str] = {
    "desk":        "desk",
    "product":     "product_code",
    "currency":    "currency_code",
    "asset_class": "asset_class",
    "cluster":     "cluster_id",
}

_GROUP_BY_LABEL: Dict[str, str] = {
    "desk":        "Desk",
    "product":     "Product",
    "currency":    "Currency",
    "asset_class": "Asset class",
    "cluster":     "Cluster",
}

_CLEAR_BTN_VISIBLE = {}
_CLEAR_BTN_HIDDEN = {"display": "none"}

_FOCUS_CHIP_VISIBLE_STYLE: Dict[str, Any] = {"display": "inline-flex"}
_FOCUS_CHIP_HIDDEN_STYLE:  Dict[str, Any] = {"display": "none"}


def register(app: "Dash", backend: "RadeBackend") -> None:
    _register_sync_groupby(app)
    _register_sync_scatter_focus(app)
    _register_render_aggregate(app, backend)
    _register_render_grouped(app, backend)
    _register_hydrate_groupby(app)


# ══════════════════════════════════════════════════════════════════════
# 1. Group-by Select + Clear → session
# ══════════════════════════════════════════════════════════════════════


def _register_sync_groupby(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],         "data", allow_duplicate=True),
        Input(PORTFOLIO_IDS["groupby_select"],     "value"),
        Input(PORTFOLIO_IDS["groupby_clear_btn"],  "n_clicks"),
        State(SHELL_IDS["session_store"],          "data"),
        prevent_initial_call=True,
    )
    def _sync_groupby(
        selected_value: Optional[str],
        clear_clicks:   Optional[int],
        session_data:   Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        current = session.evaluation.portfolio_group_by

        if trigger == PORTFOLIO_IDS["groupby_clear_btn"]:
            if not clear_clicks:
                raise PreventUpdate
            new_value: Optional[str] = None
        else:
            new_value = selected_value if selected_value else None

        if new_value == current and session.evaluation.portfolio_scatter_focus is None:
            raise PreventUpdate

        session.evaluation.portfolio_group_by = new_value
        session.evaluation.portfolio_scatter_focus = None
        return session.to_store()


# ══════════════════════════════════════════════════════════════════════
# 2. Scatter clickData / dblclick / chip-close → session focus value
# ══════════════════════════════════════════════════════════════════════


def _register_sync_scatter_focus(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],                    "data", allow_duplicate=True),
        Input(PORTFOLIO_IDS["pred_actual_scatter"],           "clickData"),
        Input(PORTFOLIO_IDS["pred_actual_scatter"],           "relayoutData"),
        Input(PORTFOLIO_IDS["focus_chip_clear_btn"],          "n_clicks"),
        State(SHELL_IDS["session_store"],                     "data"),
        prevent_initial_call=True,
    )
    def _sync_scatter_focus(
        click_data:    Optional[Dict[str, Any]],
        relayout_data: Optional[Dict[str, Any]],
        clear_clicks:  Optional[int],
        session_data:  Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        current = session.evaluation.portfolio_scatter_focus

        if session.evaluation.portfolio_group_by is None:
            if current is None:
                raise PreventUpdate
            session.evaluation.portfolio_scatter_focus = None
            return session.to_store()

        new_focus: Optional[str] = current

        if trigger == PORTFOLIO_IDS["focus_chip_clear_btn"]:
            if not clear_clicks:
                raise PreventUpdate
            new_focus = None

        elif trigger == PORTFOLIO_IDS["pred_actual_scatter"]:
            trigger_prop = ctx.triggered[0]["prop_id"].split(".")[-1] if ctx.triggered else ""
            if trigger_prop == "clickData":
                new_focus = _extract_group_from_click(click_data)
                if new_focus is None:
                    raise PreventUpdate
            elif trigger_prop == "relayoutData":
                if not _is_autorange_reset(relayout_data):
                    raise PreventUpdate
                new_focus = None
            else:
                raise PreventUpdate

        if new_focus == current:
            raise PreventUpdate

        session.evaluation.portfolio_scatter_focus = new_focus
        return session.to_store()


def _extract_group_from_click(
    click_data: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not click_data:
        return None
    points = click_data.get("points") or []
    if not points:
        return None
    cd = points[0].get("customdata")
    if not cd or not isinstance(cd, (list, tuple)) or not cd:
        return None
    value = cd[0]
    if not isinstance(value, str) or not value:
        return None
    return value


def _is_autorange_reset(relayout_data: Optional[Dict[str, Any]]) -> bool:
    if not relayout_data:
        return False
    return any(
        key.endswith(".autorange") and value is True
        for key, value in relayout_data.items()
    )


# ══════════════════════════════════════════════════════════════════════
# 3. Aggregate render — KPIs + PnL + error over time
# ══════════════════════════════════════════════════════════════════════


def _register_render_aggregate(app: "Dash", backend: "RadeBackend") -> None:
    @app.callback(
        Output(PORTFOLIO_IDS["kpi_mae_value"],      "children"),
        Output(PORTFOLIO_IDS["kpi_rmse_value"],     "children"),
        Output(PORTFOLIO_IDS["kpi_hit_rate_value"], "children"),
        Output(PORTFOLIO_IDS["kpi_coverage_value"], "children"),
        Output(PORTFOLIO_IDS["pnl_chart"],          "figure"),
        Output(PORTFOLIO_IDS["error_ts_chart"],     "figure"),
        Input(SHELL_IDS["url"],                     "pathname"),
        Input(SHELL_IDS["session_store"],           "data"),
    )
    def _render_aggregate(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, str, str, go.Figure, go.Figure]:
        if pathname != _PORTFOLIO_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        split = session.split
        filters = session.evaluation.filters

        portfolio_df = _aggregate_portfolio_frame(backend, split, filters)
        if portfolio_df is None or portfolio_df.empty:
            empty_fig = empty_figure("No portfolio data for the active filter set.")
            return (
                _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER,
                empty_fig,
                empty_figure("No error-over-time data for the active filter set."),
            )

        mae_txt, rmse_txt, hit_txt, cov_txt = _kpi_strings(portfolio_df)
        pnl_fig = portfolio_pnl(portfolio_df)
        err_fig = error_over_time(portfolio_df)

        return mae_txt, rmse_txt, hit_txt, cov_txt, pnl_fig, err_fig


# ══════════════════════════════════════════════════════════════════════
# 4. Grouped render — violin + scatter + leaderboard + focus chip
# ══════════════════════════════════════════════════════════════════════


def _register_render_grouped(app: "Dash", backend: "RadeBackend") -> None:
    @app.callback(
        Output(PORTFOLIO_IDS["residual_violin"],         "figure"),
        Output(PORTFOLIO_IDS["pred_actual_scatter"],     "figure"),
        Output(PORTFOLIO_IDS["leaderboard_grid"],        "rowData"),
        Output(PORTFOLIO_IDS["leaderboard_grid"],        "columnDefs"),
        Output(PORTFOLIO_IDS["leaderboard_empty"],       "style"),
        Output(PORTFOLIO_IDS["leaderboard_grid_wrap"],   "style"),
        Output(PORTFOLIO_IDS["leaderboard_header"],      "children"),
        Output(PORTFOLIO_IDS["focus_chip_container"],    "style"),
        Output(PORTFOLIO_IDS["focus_chip_label"],        "children"),
        Output(PORTFOLIO_IDS["groupby_clear_btn"],       "style"),
        Output(PORTFOLIO_IDS["groupby_count_label"],     "children"),
        Input(SHELL_IDS["url"],                          "pathname"),
        Input(SHELL_IDS["session_store"],                "data"),
    )
    def _render_grouped(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        if pathname != _PORTFOLIO_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        split = session.split
        filters = session.evaluation.filters
        group_by = session.evaluation.portfolio_group_by
        focus_group = session.evaluation.portfolio_scatter_focus

        per_cluster = _per_cluster_frame(backend, split, filters)
        available_groups: List[str] = []
        group_column_name: Optional[str] = None
        if group_by is not None:
            group_column_name = _GROUP_BY_COLUMN.get(group_by)
            if group_column_name and per_cluster is not None and group_column_name in per_cluster.columns:
                available_groups = sorted(
                    {str(v) for v in per_cluster[group_column_name].dropna().unique()}
                )

        effective_focus = (
            focus_group if focus_group in available_groups else None
        )

        # ── Violin + scatter ────────────────────────────────────────
        if per_cluster is None or per_cluster.empty:
            violin_fig = empty_figure("No per-cluster residuals for this filter set.")
            scatter_fig = empty_figure("No per-cluster pairs for this filter set.")
        else:
            residuals = (
                per_cluster["predicted"] - per_cluster["actual"]
            ).to_numpy(dtype=float)
            hover_labels = per_cluster.get(
                "cluster_id",
                pd.Series([""] * len(per_cluster)),
            ).astype(str).tolist()

            if group_by is None or group_column_name is None or group_column_name not in per_cluster.columns:
                violin_fig = residual_violin(residuals)
                scatter_fig = pred_actual_scatter(
                    per_cluster["predicted"].to_numpy(dtype=float),
                    per_cluster["actual"].to_numpy(dtype=float),
                    hover_labels=hover_labels,
                )
            else:
                group_vals = per_cluster[group_column_name].astype(str).tolist()
                violin_fig = residual_violin(
                    residuals,
                    group_values=group_vals,
                    group_order=available_groups,
                    group_label=_GROUP_BY_LABEL.get(group_by, group_by),
                )
                scatter_fig = pred_actual_scatter(
                    per_cluster["predicted"].to_numpy(dtype=float),
                    per_cluster["actual"].to_numpy(dtype=float),
                    group_values=group_vals,
                    group_order=available_groups,
                    focus_group=effective_focus,
                    hover_labels=hover_labels,
                )

        # ── Leaderboard ─────────────────────────────────────────────
        if group_by is None:
            leaderboard_rows: List[Dict[str, Any]] = []
            column_defs = _leaderboard_column_defs(group_by=None)
            empty_state_style: Dict[str, Any] = {}
            grid_wrap_style:   Dict[str, Any] = {"display": "none"}
        elif per_cluster is None or per_cluster.empty or group_column_name not in (per_cluster.columns if per_cluster is not None else []):
            leaderboard_rows = []
            column_defs = _leaderboard_column_defs(group_by=group_by)
            empty_state_style = {}
            grid_wrap_style = {"display": "none"}
        else:
            leaderboard_rows = _leaderboard_rows(
                per_cluster,
                group_column=group_column_name,
            )
            column_defs = _leaderboard_column_defs(group_by=group_by)
            empty_state_style = {"display": "none"}
            grid_wrap_style = {}

        header_children = _leaderboard_header_children(
            group_by=group_by,
            n_groups=len(available_groups),
            n_clusters=0 if per_cluster is None else len(per_cluster.drop_duplicates("cluster_id")) if "cluster_id" in (per_cluster.columns if per_cluster is not None else []) else 0,
        )
        focus_chip_style = (
            _FOCUS_CHIP_VISIBLE_STYLE if effective_focus is not None
            else _FOCUS_CHIP_HIDDEN_STYLE
        )
        focus_chip_label = f"Focused: {effective_focus}" if effective_focus else "Focused: —"

        clear_btn_style = (
            _CLEAR_BTN_VISIBLE if group_by is not None else _CLEAR_BTN_HIDDEN
        )
        if group_by is None:
            count_label = ""
        elif not available_groups:
            count_label = "no groups"
        else:
            count_label = f"{len(available_groups)} group{'' if len(available_groups) == 1 else 's'}"

        return (
            violin_fig,
            scatter_fig,
            leaderboard_rows,
            column_defs,
            empty_state_style,
            grid_wrap_style,
            header_children,
            focus_chip_style,
            focus_chip_label,
            clear_btn_style,
            count_label,
        )


# ══════════════════════════════════════════════════════════════════════
# 5. Hydrate Select on entry to /evaluation/portfolio
# ══════════════════════════════════════════════════════════════════════


def _register_hydrate_groupby(app: "Dash") -> None:
    @app.callback(
        Output(PORTFOLIO_IDS["groupby_select"], "value", allow_duplicate=True),
        Input(SHELL_IDS["url"],                 "pathname"),
        State(SHELL_IDS["session_store"],       "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _hydrate_groupby(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Any:
        if pathname != _PORTFOLIO_PATH:
            raise PreventUpdate
        session = Session.from_store(session_data)
        return session.evaluation.portfolio_group_by


# ══════════════════════════════════════════════════════════════════════
# Data plumbing helpers
# ══════════════════════════════════════════════════════════════════════


def _aggregate_portfolio_frame(
    backend: "RadeBackend",
    split:   str,
    filters: EvaluationFilters,
) -> Optional[pd.DataFrame]:
    df: Optional[pd.DataFrame]

    if filters.is_empty() or _only_date_filter(filters):
        res = backend.portfolio_df(split)
        if not _ok(res):
            logger.warning("portfolio_df(%s) failed: %s", split, res.error)  # type: ignore[union-attr]
            return None
        df = res.data.copy()  # type: ignore[union-attr]
    else:
        per_cluster = _per_cluster_frame(backend, split, filters)
        if per_cluster is None or per_cluster.empty:
            return None
        df = _aggregate_clusters_to_portfolio(per_cluster)

    if df is None or df.empty:
        return None
    return _apply_date_filter(df, filters)


def _per_cluster_frame(
    backend: "RadeBackend",
    split:   str,
    filters: EvaluationFilters,
) -> Optional[pd.DataFrame]:
    res_clusters = backend.clusters_df()
    if not _ok(res_clusters):
        logger.warning("clusters_df failed: %s", res_clusters.error)  # type: ignore[union-attr]
        return None
    clusters = res_clusters.data  # type: ignore[union-attr]
    if clusters is None or clusters.empty:
        return None

    masked = _apply_attribute_filters(clusters, filters)
    if masked.empty:
        return masked

    res_ts = backend.cluster_timeseries_df(split)
    if not _ok(res_ts):
        logger.warning("cluster_timeseries_df(%s) failed: %s", split, res_ts.error)  # type: ignore[union-attr]
        return None
    ts = res_ts.data  # type: ignore[union-attr]
    if ts is None or ts.empty:
        return ts

    allowed_ids = set(masked["cluster_id"].astype(str))
    ts_filtered = ts[ts["cluster_id"].astype(str).isin(allowed_ids)].copy()

    attr_cols = [
        c for c in ("desk", "product_code", "currency_code", "asset_class")
        if c in masked.columns
    ]
    if attr_cols:
        ts_filtered = ts_filtered.merge(
            masked[["cluster_id", *attr_cols]],
            on="cluster_id",
            how="left",
        )

    return _apply_date_filter(ts_filtered, filters)


def _aggregate_clusters_to_portfolio(per_cluster: pd.DataFrame) -> pd.DataFrame:
    if per_cluster.empty:
        return per_cluster

    group_keys = [c for c in ("scenario_idx", "scenario_label") if c in per_cluster.columns]
    if not group_keys:
        group_keys = ["scenario_idx"] if "scenario_idx" in per_cluster.columns else ["scenario_label"]

    agg = (
        per_cluster.groupby(group_keys, as_index=False)
        .agg(
            predicted=("predicted", "sum"),
            actual=("actual", "sum"),
        )
    )
    agg["error"] = agg["predicted"] - agg["actual"]
    agg["abs_error"] = agg["error"].abs()
    agg["squared_error"] = agg["error"] ** 2
    return agg


def _apply_attribute_filters(
    clusters: pd.DataFrame,
    filters:  EvaluationFilters,
) -> pd.DataFrame:
    df = clusters
    if filters.asset_class and "asset_class" in df.columns:
        df = df[df["asset_class"].astype(str).isin(filters.asset_class)]
    if filters.currency and "currency_code" in df.columns:
        df = df[df["currency_code"].astype(str).isin(filters.currency)]
    if filters.desk and "desk" in df.columns:
        df = df[df["desk"].astype(str).isin(filters.desk)]
    if filters.product and "product_code" in df.columns:
        df = df[df["product_code"].astype(str).isin(filters.product)]
    return df


def _apply_date_filter(
    df:       pd.DataFrame,
    filters:  EvaluationFilters,
) -> pd.DataFrame:
    if "scenario_label" not in df.columns:
        return df
    out = df
    if filters.date_from:
        out = out[out["scenario_label"].astype(str) >= str(filters.date_from)]
    if filters.date_to:
        out = out[out["scenario_label"].astype(str) <= str(filters.date_to)]
    return out


def _only_date_filter(filters: EvaluationFilters) -> bool:
    return (
        not filters.asset_class
        and not filters.currency
        and not filters.desk
        and not filters.product
        and (filters.date_from is not None or filters.date_to is not None)
    )


def _ok(res: "BackendResult[Any]") -> bool:
    return bool(getattr(res, "ok", False)) and getattr(res, "data", None) is not None


# ══════════════════════════════════════════════════════════════════════
# KPI + leaderboard computations
# ══════════════════════════════════════════════════════════════════════


def _kpi_strings(portfolio_df: pd.DataFrame) -> Tuple[str, str, str, str]:
    if portfolio_df is None or portfolio_df.empty:
        return _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER

    abs_err = (
        portfolio_df["abs_error"]
        if "abs_error" in portfolio_df.columns
        else (portfolio_df["predicted"] - portfolio_df["actual"]).abs()
    ).astype(float)
    sq_err = (
        portfolio_df["squared_error"]
        if "squared_error" in portfolio_df.columns
        else (portfolio_df["predicted"] - portfolio_df["actual"]) ** 2
    ).astype(float)

    mae_val = float(abs_err.mean()) if not abs_err.empty else float("nan")
    rmse_val = float(np.sqrt(sq_err.mean())) if not sq_err.empty else float("nan")

    hit_mask = np.sign(portfolio_df["predicted"]) == np.sign(portfolio_df["actual"])
    n_total = int(hit_mask.size)
    hit_pct = float(hit_mask.mean() * 100.0) if n_total > 0 else float("nan")

    finite_mask = (
        portfolio_df["predicted"].notna()
        & portfolio_df["actual"].notna()
        & np.isfinite(portfolio_df["predicted"])
        & np.isfinite(portfolio_df["actual"])
    )
    coverage_pct = float(finite_mask.mean() * 100.0) if n_total > 0 else float("nan")

    return (
        _fmt_num(mae_val, digits=4),
        _fmt_num(rmse_val, digits=4),
        _fmt_pct(hit_pct),
        _fmt_pct(coverage_pct),
    )


def _leaderboard_rows(
    per_cluster:   pd.DataFrame,
    *,
    group_column:  str,
) -> List[Dict[str, Any]]:
    if per_cluster is None or per_cluster.empty or group_column not in per_cluster.columns:
        return []

    df = per_cluster.copy()
    df["_abs_err"] = (df["predicted"] - df["actual"]).abs()
    df["_sq_err"]  = (df["predicted"] - df["actual"]) ** 2
    df["_hit"]     = (np.sign(df["predicted"]) == np.sign(df["actual"])).astype(float)

    agg = df.groupby(group_column, dropna=False).agg(
        mae=("_abs_err", "mean"),
        rmse_sq=("_sq_err", "mean"),
        hit_rate=("_hit", "mean"),
        contribution_total=("_abs_err", "sum"),
        n_clusters=("cluster_id", "nunique"),
    ).reset_index()

    total_abs = float(agg["contribution_total"].sum())
    if total_abs <= 0:
        agg["contribution"] = 0.0
    else:
        agg["contribution"] = agg["contribution_total"] / total_abs * 100.0

    agg["rmse"] = np.sqrt(agg["rmse_sq"].astype(float))
    agg["hit_rate"] = agg["hit_rate"].astype(float) * 100.0

    agg = agg.sort_values("contribution", ascending=False)

    rows: List[Dict[str, Any]] = []
    for _, r in agg.iterrows():
        rows.append(
            {
                "group_label":   "" if pd.isna(r[group_column]) else str(r[group_column]),
                "mae":           _round(r["mae"], 4),
                "rmse":          _round(r["rmse"], 4),
                "hit_rate":      _round(r["hit_rate"], 1),
                "contribution":  _round(r["contribution"], 1),
                "n_clusters":    int(r["n_clusters"]),
            }
        )
    return rows


def _leaderboard_column_defs(*, group_by: Optional[str]) -> List[Dict[str, Any]]:
    first_label = _GROUP_BY_LABEL.get(group_by or "", "Break-down") if group_by else "Break-down"
    return [
        {"field": "group_label", "headerName": first_label, "flex": 2, "minWidth": 140},
        {
            "field": "mae",
            "headerName": "MAE",
            "flex": 1,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.4f')(params.value)"},
        },
        {
            "field": "rmse",
            "headerName": "RMSE",
            "flex": 1,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.4f')(params.value)"},
        },
        {
            "field": "hit_rate",
            "headerName": "Hit %",
            "flex": 1,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.1f')(params.value)"},
        },
        {
            "field": "contribution",
            "headerName": "Contribution %",
            "flex": 1,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.1f')(params.value)"},
        },
        {"field": "n_clusters", "headerName": "Clusters", "flex": 1, "type": "numericColumn"},
    ]


def _leaderboard_header_children(
    *,
    group_by:   Optional[str],
    n_groups:   int,
    n_clusters: int,
) -> List[Any]:
    from dash import html

    if group_by is None:
        title = "Leaderboard"
        subtitle = "Pick a break-down dimension to compare contributors."
    else:
        title = f"Leaderboard — by {_GROUP_BY_LABEL.get(group_by, group_by)}"
        subtitle = (
            f"{n_groups} group{'' if n_groups == 1 else 's'} "
            f"covering {n_clusters} cluster{'' if n_clusters == 1 else 's'} "
            "under the active filter set."
        )

    return [
        html.Div(
            className="flex flex-col",
            children=[
                html.Div(title,    className="text-sm font-semibold text-slate-200"),
                html.Div(subtitle, className="text-xs text-slate-500"),
            ],
        ),
    ]


def _fmt_num(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return _PLACEHOLDER
    try:
        val = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val) or not np.isfinite(val):
        return _PLACEHOLDER
    return f"{val:,.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return _PLACEHOLDER
    try:
        val = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val) or not np.isfinite(val):
        return _PLACEHOLDER
    return f"{val:,.1f}%"


def _round(value: Any, digits: int) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val) or not np.isfinite(val):
        return None
    return round(val, digits)


_ = no_update  # preserved for future callbacks in this module


__all__ = ["register"]
```

---

### Step 9 · EDIT `src/ui/apps/rade_analytics/callbacks/__init__.py`

> **Action.**  Two-line edit — import `portfolio_cb` and call its
> `register()` from `register_all`.  Safe to paste the whole file.

```python
"""Callback orchestration — one public entry point: :func:`register_all`."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..router import register_router
from . import evaluation_cb, overview_cb, portfolio_cb, splash_cb

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


def register_all(app: "Dash", backend: "RadeBackend") -> None:
    register_router(app, backend)
    splash_cb.register(app, backend)
    overview_cb.register(app, backend)
    evaluation_cb.register(app, backend)
    portfolio_cb.register(app, backend)


__all__ = ["register_all"]
```

---

### Step 10 · CREATE `examples/rade_analytics/_mock_backend.py`

> **Action.**  Create a new shared helper file under
> `examples/rade_analytics/`.  Both the Overview (05) and Portfolio
> (06) preview scripts import from here so the synthetic dataset stays
> consistent between pages.

```python
"""Shared synthetic :class:`RadeBackend` for every preview script."""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.rade_ml_pt.ensemble.api.models.clusters import (
    ClusterInfo,
    ClustersResponse,
)
from src.rade_ml_pt.ensemble.api.models.meta import (
    HealthResponse,
    VersionsResponse,
)
from src.ui.apps.rade_analytics.data.backend import BackendResult, RadeBackend


_ASSET_CLASSES = ("rates", "fx", "credit", "equity")
_CURRENCIES    = ("USD", "EUR", "GBP", "JPY")
_DESKS         = ("Alpha", "Beta", "Gamma")
_PRODUCTS      = ("swap", "option", "forward", "bond")
_N_SCENARIOS   = 48


class MockRadeBackend(RadeBackend):
    """Deterministic in-memory backend for Dash UI preview scripts."""

    def __init__(self, *, n_clusters: int = 12, seed: int = 42) -> None:
        # NOT calling super().__init__ on purpose — the parent
        # constructor requires a client and cache which we don't have.
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._seed = seed
        self._n_clusters = n_clusters
        self._cluster_ids: List[str] = [
            f"cluster_{i + 1:02d}" for i in range(n_clusters)
        ]
        self._versions: List[str] = [
            "v2026.04.17-a1b2c",
            "v2026.04.10-f3e4d7",
            "v2026.04.03-9876a2",
        ]
        self._cluster_attrs: Dict[str, Dict[str, Any]] = {
            cid: {
                "asset_class":   self._rng.choice(_ASSET_CLASSES),
                "currency_code": self._rng.choice(_CURRENCIES),
                "desk":          self._rng.choice(_DESKS),
                "product_code":  self._rng.choice(_PRODUCTS),
            }
            for cid in self._cluster_ids
        }
        raw_weights = self._np_rng.uniform(0.2, 1.8, size=n_clusters)
        self._cluster_weights: Dict[str, float] = dict(
            zip(self._cluster_ids, raw_weights / raw_weights.sum() * n_clusters)
        )

    # ─── Meta endpoints ─────────────────────────────────────────────

    def health(self) -> BackendResult[HealthResponse]:
        return BackendResult.success(
            HealthResponse(
                status="ok",
                version=self._versions[0],
                artifacts_dir="/tmp/mock-artifacts",
            )
        )

    def versions(self) -> BackendResult[VersionsResponse]:
        return BackendResult.success(
            VersionsResponse(
                active=self._versions[0],
                available=list(self._versions),
            )
        )

    # ─── Metrics endpoints ──────────────────────────────────────────

    def ensemble_metrics_df(self) -> BackendResult[pd.DataFrame]:
        rows: List[Dict[str, Any]] = []
        for split, scale in (("train", 0.9), ("val", 1.05), ("test", 1.15)):
            mae = 0.0012 * scale
            mse = (mae ** 2) * 1.4
            rmse = float(np.sqrt(mse))
            rows.append(
                {
                    "split":  split,
                    "mae":    mae,
                    "mse":    mse,
                    "rmse":   rmse,
                    "max_ae": mae * 18.0,
                    "p95_ae": mae * 4.5,
                    "p99_ae": mae * 9.0,
                }
            )
        return BackendResult.success(pd.DataFrame(rows))

    def per_member_metrics_df(
        self,
        *,
        split:      Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        splits = [split] if split else ["train", "val", "test"]
        frames: List[pd.DataFrame] = []
        for s in splits:
            ts = self._cluster_timeseries(s)
            if cluster_id:
                ts = ts[ts["cluster_id"] == cluster_id]
            grouped = (
                ts.assign(
                    _abs_err=ts["abs_error"],
                    _sq_err=ts["squared_error"],
                )
                .groupby("cluster_id", as_index=False)
                .agg(
                    mae=("_abs_err", "mean"),
                    rmse_sq=("_sq_err", "mean"),
                    max_ae=("_abs_err", "max"),
                    p95_ae=("_abs_err", lambda x: float(np.quantile(x, 0.95))),
                    p99_ae=("_abs_err", lambda x: float(np.quantile(x, 0.99))),
                    n_scenarios=("scenario_idx", "nunique"),
                )
            )
            grouped["rmse"] = np.sqrt(grouped["rmse_sq"].astype(float))
            grouped["mse"] = grouped["rmse"] ** 2
            grouped["split"] = s
            grouped["n_targets"] = [
                self._rng.randint(120, 480) for _ in range(len(grouped))
            ]
            frames.append(
                grouped[
                    [
                        "cluster_id", "split", "mae", "mse", "rmse",
                        "max_ae", "p95_ae", "p99_ae",
                        "n_targets", "n_scenarios",
                    ]
                ]
            )
        if not frames:
            return BackendResult.success(pd.DataFrame())
        return BackendResult.success(pd.concat(frames, ignore_index=True))

    # ─── Cluster endpoints ──────────────────────────────────────────

    def clusters(
        self, *, cluster_id: Optional[str] = None,
    ) -> BackendResult[ClustersResponse]:
        entries: List[ClusterInfo] = []
        for cid in self._cluster_ids:
            if cluster_id and cid != cluster_id:
                continue
            n_trades = 80 + int(self._cluster_weights[cid] * 180)
            entries.append(
                ClusterInfo(
                    cluster_id=cid,
                    n_trades=n_trades,
                    attributes=dict(self._cluster_attrs[cid]),
                )
            )
        return BackendResult.success(
            ClustersResponse(
                clusters=entries,
                attribute_names=["asset_class", "currency_code", "desk", "product_code"],
            )
        )

    def clusters_df(
        self, *, cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        res = self.clusters(cluster_id=cluster_id)
        if not res.ok or res.data is None:
            return BackendResult.failure(
                error=res.error or "",
                status_code=res.status_code,
            )
        rows = [
            {"cluster_id": c.cluster_id, "n_trades": c.n_trades, **c.attributes}
            for c in res.data.clusters
        ]
        return BackendResult.success(pd.DataFrame(rows))

    # ─── Portfolio + cluster timeseries ────────────────────────────

    def portfolio_df(self, split: str) -> BackendResult[pd.DataFrame]:
        ts = self._cluster_timeseries(split)
        if ts.empty:
            return BackendResult.success(ts)
        agg = (
            ts.groupby(["scenario_idx", "scenario_label"], as_index=False)
            .agg(
                predicted=("predicted", "sum"),
                actual=("actual", "sum"),
            )
        )
        agg["error"] = agg["predicted"] - agg["actual"]
        agg["abs_error"] = agg["error"].abs()
        agg["squared_error"] = agg["error"] ** 2
        return BackendResult.success(agg)

    def cluster_timeseries_df(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        df = self._cluster_timeseries(split)
        if cluster_id:
            df = df[df["cluster_id"] == cluster_id]
        return BackendResult.success(df.reset_index(drop=True))

    # ─── Internal — seeded synthetic data generators ────────────────

    def _cluster_timeseries(self, split: str) -> pd.DataFrame:
        return _cluster_timeseries_cached(
            mock_id=id(self),
            split=split,
            seed=self._seed,
            cluster_ids=tuple(self._cluster_ids),
            cluster_weights=tuple(
                (cid, self._cluster_weights[cid]) for cid in self._cluster_ids
            ),
        )


@lru_cache(maxsize=64)
def _cluster_timeseries_cached(
    *,
    mock_id:          int,
    split:            str,
    seed:             int,
    cluster_ids:      tuple[str, ...],
    cluster_weights:  tuple[tuple[str, float], ...],
) -> pd.DataFrame:
    del mock_id  # cache key only
    rng = np.random.default_rng(seed + hash(split) % 10_000)
    labels = pd.date_range("2025-11-01", periods=_N_SCENARIOS, freq="D").strftime(
        "%Y-%m-%d"
    )
    noise_scale = {"train": 0.008, "val": 0.012, "test": 0.018}.get(split, 0.012)

    weights = dict(cluster_weights)
    frames: List[pd.DataFrame] = []
    for cid in cluster_ids:
        w = float(weights[cid])
        actual_inc = rng.normal(0.003 * w, 0.004 * max(w, 0.3), _N_SCENARIOS)
        actual = np.cumsum(actual_inc) + 0.08 * w
        pred = actual + rng.normal(0, noise_scale * max(w, 0.3), _N_SCENARIOS)
        error = pred - actual
        frames.append(
            pd.DataFrame(
                {
                    "cluster_id":     cid,
                    "scenario_idx":   list(range(_N_SCENARIOS)),
                    "scenario_label": labels,
                    "predicted":      pred,
                    "actual":         actual,
                    "error":          error,
                    "abs_error":      np.abs(error),
                    "squared_error":  error ** 2,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


__all__ = ["MockRadeBackend"]
```

---

### Step 11 · REPLACE `examples/rade_analytics/05_overview_preview_live.py`

> **Action.**  Replace the existing file — the old version defined its
> own mock inline; this one imports the shared one from Step 10.

```python
"""End-to-end smoke test for the Overview page with *no* real backend."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import dash._dash_renderer  # noqa: E402
dash._dash_renderer._set_react_version("18.2.0")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dash import Dash  # noqa: E402

from _mock_backend import MockRadeBackend  # noqa: E402  (sibling import)
from src.ui.apps.rade_analytics.callbacks import register_all  # noqa: E402
from src.ui.apps.rade_analytics.config import RadeUiSettings, set_settings  # noqa: E402
from src.ui.apps.rade_analytics.layouts import (  # noqa: E402
    INDEX_STRING,
    META_TAGS,
    build_shell,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rade.preview.overview_live")


def build_preview_app() -> Dash:
    settings = RadeUiSettings(
        api_url="http://mock",
        cache_type="NullCache",
        debug=True,
    )
    set_settings(settings)

    assets_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "ui" / "apps" / "rade_analytics" / "assets"
    )

    app = Dash(
        __name__,
        title="Rade — Overview live preview (mock)",
        update_title=None,
        index_string=INDEX_STRING,
        meta_tags=META_TAGS,
        assets_folder=str(assets_dir),
        assets_ignore=r"tailwind\.(config\.js|input\.css)|README\.md",
        suppress_callback_exceptions=True,
    )

    backend = MockRadeBackend()
    app.server.config["rade_backend"] = backend
    app.server.config["rade_settings"] = settings

    app.layout = build_shell()
    register_all(app, backend)

    log.info("Preview ready — landing on / with MockRadeBackend")
    return app


if __name__ == "__main__":
    build_preview_app().run(
        debug=True,
        host="0.0.0.0",
        port=8052,
    )
```

---

### Step 12 · CREATE `examples/rade_analytics/06_portfolio_preview_live.py`

> **Action.**  Create the file — the Portfolio smoke-test twin of 05,
> pointing at port 8053 so both previews can coexist.

```python
"""End-to-end smoke test for the Evaluation → Portfolio sub-tab."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import dash._dash_renderer  # noqa: E402
dash._dash_renderer._set_react_version("18.2.0")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dash import Dash  # noqa: E402

from _mock_backend import MockRadeBackend  # noqa: E402  (sibling import)
from src.ui.apps.rade_analytics.callbacks import register_all  # noqa: E402
from src.ui.apps.rade_analytics.config import RadeUiSettings, set_settings  # noqa: E402
from src.ui.apps.rade_analytics.layouts import (  # noqa: E402
    INDEX_STRING,
    META_TAGS,
    build_shell,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rade.preview.portfolio_live")


def build_preview_app() -> Dash:
    settings = RadeUiSettings(
        api_url="http://mock",
        cache_type="NullCache",
        debug=True,
    )
    set_settings(settings)

    assets_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "ui" / "apps" / "rade_analytics" / "assets"
    )

    app = Dash(
        __name__,
        title="Rade — Portfolio live preview (mock)",
        update_title=None,
        index_string=INDEX_STRING,
        meta_tags=META_TAGS,
        assets_folder=str(assets_dir),
        assets_ignore=r"tailwind\.(config\.js|input\.css)|README\.md",
        suppress_callback_exceptions=True,
    )

    backend = MockRadeBackend()
    app.server.config["rade_backend"] = backend
    app.server.config["rade_settings"] = settings

    app.layout = build_shell()
    register_all(app, backend)

    log.info(
        "Preview ready — open http://localhost:8053/evaluation/portfolio"
    )
    return app


if __name__ == "__main__":
    build_preview_app().run(
        debug=True,
        host="0.0.0.0",
        port=8053,
    )
```

---

### Done — verification checklist

- [ ] `python -m py_compile src/ui/apps/rade_analytics/data/session.py src/ui/apps/rade_analytics/figures/*.py src/ui/apps/rade_analytics/layouts/evaluation/portfolio.py src/ui/apps/rade_analytics/callbacks/portfolio_cb.py src/ui/apps/rade_analytics/callbacks/__init__.py examples/rade_analytics/_mock_backend.py examples/rade_analytics/05_overview_preview_live.py examples/rade_analytics/06_portfolio_preview_live.py` prints nothing (no syntax errors).
- [ ] `python examples/rade_analytics/06_portfolio_preview_live.py` starts and logs the ready line.
- [ ] Browser at `http://localhost:8053/evaluation/portfolio` shows:
      * 4 KPI cards filled with numbers,
      * a purple predicted / dashed actual line chart,
      * a rose rolling-error band,
      * empty violin + scatter + leaderboard placeholders until you
        pick a break-down dimension.
- [ ] Pick "Desk" — violin fills with 3 side-by-side violins, scatter
      colours points per desk, leaderboard lists one row per desk.
- [ ] Click a point in the scatter — chip appears ("Focused: Alpha
      [× Show all]"), scatter narrows to that desk only, violin &
      leaderboard stay at full coverage.
- [ ] Double-click the scatter plot → focus clears, all groups come
      back.
- [ ] Click "× Show all" on the chip → same, focus clears.
- [ ] Toggle topbar split (Train / Val / Test) → every slot redraws.

If any step misbehaves, the terminal running the preview logs warnings
starting with `rade_analytics.callbacks.portfolio_cb` — they're the
first place to look.
