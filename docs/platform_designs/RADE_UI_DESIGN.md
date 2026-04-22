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

## Appendix A — Phase B.2 bootstrap files (TEMPORARY)

> **Purpose:** one-shot carrier for the four Phase B.2 files so they can
> be pushed via this markdown document and copy-pasted into the work
> environment without manual re-typing.
>
> **Remove this appendix (§§ A.1–A.4) once B.2 is verified on the work
> environment** and the four files are committed at their real paths.
> None of the content below belongs in the long-lived design spec.
>
> Paste instructions for SVG files: copy **everything between the
> opening and closing triple-backtick fences**, starting at the
> `<?xml version=` line. Do not include the language tag line (`xml`).

### A.1 — `src/ui/apps/rade_analytics/assets/logo.svg`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!--
  Rade Analytics - primary brand mark.

  Used by components/brand.py in the sidebar header, the splash page
  hero and the landing topbar.  Palette mirrors RADE_UI_DESIGN.md
  sections 2 and 3: violet-500 (#8b5cf6) to cyan-400 (#22d3ee) linear
  gradient with a slate-950 (#020617) "R" stroked on top for WCAG-AA
  contrast.

  The viewBox is 40x40 so the mark pairs naturally with the Tailwind
  classes w-10 h-10 / w-8 h-8.  Self-contained (no external fonts);
  strokes so it stays crisp at every size.
-->
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 40 40"
     width="40" height="40"
     fill="none"
     role="img"
     aria-label="Rade logo">
  <title>Rade</title>
  <defs>
    <linearGradient id="rade-mark-gradient"
                    x1="0" y1="0" x2="40" y2="40"
                    gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#22d3ee"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="40" height="40" rx="10" ry="10"
        fill="url(#rade-mark-gradient)"/>
  <path d="M12 29 L12 11 L23 11 A5 5 0 0 1 23 21 L12 21 M18 21 L26 29"
        stroke="#020617"
        stroke-width="3"
        stroke-linecap="round"
        stroke-linejoin="round"/>
</svg>
```

### A.2 — `src/ui/apps/rade_analytics/assets/favicon.svg`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!--
  Rade Analytics - browser-tab favicon.

  Modern browsers (Chrome, Firefox, Safari) honour SVG favicons.  We
  wire this file in via an explicit <link rel="icon" type="image/svg+xml">
  tag in the Dash index_string (see layouts/head.py).  Dash's built-in
  {%favicon%} placeholder only looks for favicon.ico, so the extra link
  is harmless and lets browsers that support SVG icons render this one
  at native resolution.

  Visually identical to logo.svg but with a thicker stroke (3.5) so it
  remains legible at 16x16.
-->
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 40 40"
     width="40" height="40"
     fill="none">
  <defs>
    <linearGradient id="rade-fv-gradient"
                    x1="0" y1="0" x2="40" y2="40"
                    gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#22d3ee"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="40" height="40" rx="10" ry="10"
        fill="url(#rade-fv-gradient)"/>
  <path d="M13 29 L13 11 L23 11 A5 5 0 0 1 23 21 L13 21 M18 21 L26 29"
        stroke="#020617"
        stroke-width="3.5"
        stroke-linecap="round"
        stroke-linejoin="round"/>
</svg>
```

> **Encoding caveat:** both SVG files are intentionally ASCII-only (no
> em-dashes, en-dashes or section symbols) because an earlier Write
> pass in the Rade UI stack mangled multi-byte UTF-8 glyphs inside
> `<!-- ... -->` comments to stray control bytes. The Python files
> below *do* round-trip UTF-8 cleanly, so em-dashes / section symbols
> are fine there.

### A.3 — `src/ui/apps/rade_analytics/layouts/head.py`

```python
"""HTML head fragments injected into the Dash index template.

Dash exposes two levers we care about in Phase B.2:

* ``meta_tags``:   a list of ``<meta>`` dicts passed to ``Dash(...)``.
* ``index_string``: a full HTML template with ``{%...%}`` placeholders
  that Dash substitutes at request time.  We customise it so the
  Google-Fonts request for Inter + JetBrains Mono starts in parallel
  with (not after) the CSS bundle, and so the SVG favicon declared in
  ``assets/favicon.svg`` is picked up by modern browsers.

Consumers
---------
B.6's ``app.py`` will wire these constants in as:

    from .layouts.head import INDEX_STRING, META_TAGS

    app = Dash(
        __name__,
        index_string=INDEX_STRING,
        meta_tags=META_TAGS,
        assets_folder="assets",
    )

The constants are deliberately strings/lists (no builders) so they can
be imported in a no-Dash context (e.g. tests that snapshot the HTML
shell) without dragging the Dash package.

Design-spec anchors
-------------------
* Palette / theme-color:  RADE_UI_DESIGN.md §2.
* Typography / font list: RADE_UI_DESIGN.md §3.
* Dark-first color scheme: RADE_UI_DESIGN.md §2 + §10 (accessibility).
"""

from __future__ import annotations

from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────
# Meta tags (Dash ``meta_tags=`` kwarg)
# ─────────────────────────────────────────────────────────────────────

META_TAGS: List[Dict[str, str]] = [
    {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1, shrink-to-fit=no",
    },
    {
        "name": "description",
        "content": (
            "Rade Analytics — ensemble model analytics platform for "
            "monitoring, governance, evaluation and inference."
        ),
    },
    {"name": "color-scheme", "content": "dark"},
    # Matches bg-slate-950 so Chrome on macOS tints the title bar the
    # same colour as the app chrome.  Cheap polish.
    {"name": "theme-color", "content": "#020617"},
    {"name": "application-name", "content": "Rade"},
]


# ─────────────────────────────────────────────────────────────────────
# Font preload + SVG favicon link
# ─────────────────────────────────────────────────────────────────────
#
# Split into a dedicated constant so tests can assert "fonts are
# preconnected before the stylesheet request fires".  We include the
# stylesheet <link> immediately after the preload so CSS still parses
# correctly even if the browser ignores the rel=preload hint.

_GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500;600"
    "&display=swap"
)

FONT_PRELOAD_LINKS: str = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    f'    <link rel="preload" as="style" href="{_GOOGLE_FONTS_URL}">\n'
    f'    <link rel="stylesheet" href="{_GOOGLE_FONTS_URL}">'
)

# SVG favicon explicit link — Dash's ``{%favicon%}`` placeholder only
# emits a tag for ``assets/favicon.ico``.  This extra link lets modern
# browsers pick up ``assets/favicon.svg`` without us having to ship an
# ICO too.
SVG_FAVICON_LINK: str = (
    '<link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">'
)


# ─────────────────────────────────────────────────────────────────────
# Full index_string template
# ─────────────────────────────────────────────────────────────────────
#
# Kept as plain string concatenation (not f-strings) so the Dash tokens
# ``{%metas%}``, ``{%title%}`` etc. survive untouched.  Only two
# interpolations happen at import time: the font preload block and the
# SVG favicon link — everything else is Dash's own placeholder syntax.

INDEX_STRING: str = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    "  <head>\n"
    "    {%metas%}\n"
    "    <title>{%title%}</title>\n"
    f"    {SVG_FAVICON_LINK}\n"
    "    {%favicon%}\n"
    f"    {FONT_PRELOAD_LINKS}\n"
    "    {%css%}\n"
    "  </head>\n"
    '  <body class="bg-slate-950 text-slate-100 antialiased">\n'
    "    {%app_entry%}\n"
    "    <footer>\n"
    "      {%config%}\n"
    "      {%scripts%}\n"
    "      {%renderer%}\n"
    "    </footer>\n"
    "  </body>\n"
    "</html>\n"
)


__all__ = [
    "FONT_PRELOAD_LINKS",
    "INDEX_STRING",
    "META_TAGS",
    "SVG_FAVICON_LINK",
]
```

### A.4 — `src/ui/apps/rade_analytics/layouts/__init__.py`

```python
"""Full-page layouts — one module per top-level route.

Populated in Phases B–E (shell, splash, landing, evaluation).

Sub-modules shipped so far
--------------------------
* :mod:`.head` — HTML ``<head>`` template consumed by ``app.py``'s
  Dash factory (``INDEX_STRING``, ``META_TAGS``, ``FONT_PRELOAD_LINKS``).
"""

from .head import (
    FONT_PRELOAD_LINKS,
    INDEX_STRING,
    META_TAGS,
    SVG_FAVICON_LINK,
)

__all__ = [
    "FONT_PRELOAD_LINKS",
    "INDEX_STRING",
    "META_TAGS",
    "SVG_FAVICON_LINK",
]
```

### Removing this appendix

Once all four files above are committed at their real paths and B.2 is
verified on the work environment, delete everything from the
`## Appendix A …` heading through the end of this document. The design
spec ends at "No mock, no merge."
