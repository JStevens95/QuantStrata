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

## Appendix A — `assets/rade.css`

Hand-authored CSS bundle for the Rade Analytics UI.  Lives at
`src/ui/apps/rade_analytics/assets/rade.css`; Dash auto-loads every
file under `assets/` at startup, so dropping this file in place is
all that is required — no bundler, no import statement.

Contents (in source order):

1. **Preflight + scrollbar + focus ring** — modern-normalize subset
   plus the design-spec scrollbar / keyboard-focus rules.
2. **Tailwind utility subset** — every `flex`, `grid`, `text-*`,
   `bg-*`, `border-*` etc. used by the Phase B–E components.
   Compiled, not generated at runtime, so the bundle stays under
   ~40 KB unminified and the dev loop has no Tailwind step.
3. **`rade-*` component classes** — the hand-written pieces that
   would otherwise need arbitrary-value Tailwind: `.rade-card`,
   `.rade-kpi-*`, `.rade-nav-item`, `.rade-skeleton-shimmer`,
   `.rade-grid` (AG Grid theme).
4. **Splash page block** — full-viewport hero, gradient backdrops,
   v2 layout (`.rade-splash-layout` / `.rade-splash-hero`).
5. **Overview page + topbar** — breadcrumb row, page container,
   cluster heatmap, list rows, status chips, recent-activity feed,
   quick-actions footer.
6. **Evaluation page** — page container, two-row filter bar
   (`.rade-filter-bar` / `.rade-filter-row--{one,two}`), self-
   contained chip pill (`.rade-filter-chip*`) shared between the
   server-side initial render and the clientside update callback in
   `assets/js/evaluation.js`, sub-tab content slot.

Adding new utilities: prefer extending an existing `rade-*`
component class over inlining new arbitrary-value Tailwind, since
the bundle is hand-compiled and silently ignores classes we have
not pre-baked.

```css
/*!
 * Rade Analytics — compiled Tailwind bundle.
 * Source of truth: tailwind.config.js + tailwind.input.css next to this file.
 * If you need a class not present here, see assets/README.md for how to regenerate.
 *
 * Scope: every Tailwind utility the Phase B–E components use, plus the
 * custom @layer components from the input file.  Keeps the bundle under
 * ~40 KB unminified.
 */

/* ── Preflight (modern-normalize subset) ───────────────────────────── */
*, ::before, ::after {
  box-sizing: border-box;
  border-width: 0;
  border-style: solid;
  border-color: currentColor;
}
::before, ::after { --tw-content: ""; }
html {
  line-height: 1.5;
  -webkit-text-size-adjust: 100%;
  -moz-tab-size: 4;
  tab-size: 4;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, sans-serif;
  font-feature-settings: "cv02", "cv03", "cv04", "cv11";
  color-scheme: dark;
}
body {
  margin: 0;
  line-height: inherit;
  min-height: 100vh;
  background-color: #020617;
  color: #f1f5f9;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
hr { height: 0; color: inherit; border-top-width: 1px; }
abbr:where([title]) { text-decoration: underline dotted; }
h1, h2, h3, h4, h5, h6 { font-size: inherit; font-weight: inherit; margin: 0; }
a { color: inherit; text-decoration: inherit; }
b, strong { font-weight: bolder; }
code, kbd, samp, pre {
  font-family: "JetBrains Mono", "IBM Plex Mono", ui-monospace, SFMono-Regular,
    Menlo, Consolas, monospace;
  font-size: 1em;
}
small { font-size: 80%; }
sub, sup { font-size: 75%; line-height: 0; position: relative; vertical-align: baseline; }
sub { bottom: -0.25em; }
sup { top: -0.5em; }
table { text-indent: 0; border-color: inherit; border-collapse: collapse; }
button, input, optgroup, select, textarea {
  font: inherit; color: inherit; margin: 0; padding: 0;
}
button, select { text-transform: none; }
button, [type="button"], [type="reset"], [type="submit"] {
  -webkit-appearance: button; background-color: transparent; background-image: none;
}
:-moz-focusring { outline: auto; }
:-moz-ui-invalid { box-shadow: none; }
progress { vertical-align: baseline; }
::-webkit-inner-spin-button, ::-webkit-outer-spin-button { height: auto; }
[type="search"] { -webkit-appearance: textfield; outline-offset: -2px; }
::-webkit-search-decoration { -webkit-appearance: none; }
::-webkit-file-upload-button { -webkit-appearance: button; font: inherit; }
summary { display: list-item; }
blockquote, dl, dd, h1, h2, h3, h4, h5, h6, hr, figure, p, pre { margin: 0; }
fieldset { margin: 0; padding: 0; }
legend { padding: 0; }
ol, ul, menu { list-style: none; margin: 0; padding: 0; }
textarea { resize: vertical; }
input::placeholder, textarea::placeholder { opacity: 1; color: #64748b; }
button, [role="button"] { cursor: pointer; }
:disabled { cursor: default; }
img, svg, video, canvas, audio, iframe, embed, object {
  display: block; vertical-align: middle;
}
img, video { max-width: 100%; height: auto; }
[hidden] { display: none; }

/* ── Scrollbar (§design) ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgb(30 41 59 / 0.8);
  border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover { background: rgb(51 65 85 / 0.9); }

/* ── Keyboard focus ring (§10) ─────────────────────────────────────── */
:focus-visible {
  outline: none;
  box-shadow: 0 0 0 2px #020617, 0 0 0 4px #8b5cf6;
}

/* ─────────────────────────────────────────────────────────────────── */
/* UTILITIES                                                            */
/* ─────────────────────────────────────────────────────────────────── */

/* ── Display ───────────────────────────────────────────────────────── */
.block { display: block; }
.inline-block { display: inline-block; }
.inline { display: inline; }
.flex { display: flex; }
.inline-flex { display: inline-flex; }
.grid { display: grid; }
.inline-grid { display: inline-grid; }
.hidden { display: none; }

/* ── Position ──────────────────────────────────────────────────────── */
.static { position: static; }
.relative { position: relative; }
.absolute { position: absolute; }
.fixed { position: fixed; }
.sticky { position: sticky; }

.inset-0 { inset: 0; }
.inset-x-0 { left: 0; right: 0; }
.inset-y-0 { top: 0; bottom: 0; }
.top-0 { top: 0; } .right-0 { right: 0; } .bottom-0 { bottom: 0; } .left-0 { left: 0; }
.top-2 { top: 0.5rem; } .right-2 { right: 0.5rem; } .bottom-2 { bottom: 0.5rem; } .left-2 { left: 0.5rem; }
.top-4 { top: 1rem; } .right-4 { right: 1rem; } .bottom-4 { bottom: 1rem; } .left-4 { left: 1rem; }

/* ── Z-index ───────────────────────────────────────────────────────── */
.z-0 { z-index: 0; }
.z-10 { z-index: 10; }
.z-20 { z-index: 20; }
.z-30 { z-index: 30; }
.z-40 { z-index: 40; }
.z-50 { z-index: 50; }

/* ── Flex / grid ───────────────────────────────────────────────────── */
.flex-row { flex-direction: row; }
.flex-col { flex-direction: column; }
.flex-wrap { flex-wrap: wrap; }
.flex-nowrap { flex-wrap: nowrap; }
.flex-1 { flex: 1 1 0%; }
.flex-auto { flex: 1 1 auto; }
.flex-none { flex: none; }
.grow { flex-grow: 1; }
.grow-0 { flex-grow: 0; }
.shrink { flex-shrink: 1; }
.shrink-0 { flex-shrink: 0; }

.items-start { align-items: flex-start; }
.items-end { align-items: flex-end; }
.items-center { align-items: center; }
.items-baseline { align-items: baseline; }
.items-stretch { align-items: stretch; }
.justify-start { justify-content: flex-start; }
.justify-end { justify-content: flex-end; }
.justify-center { justify-content: center; }
.justify-between { justify-content: space-between; }
.justify-around { justify-content: space-around; }
.justify-evenly { justify-content: space-evenly; }

.content-start { align-content: flex-start; }
.content-center { align-content: center; }
.content-between { align-content: space-between; }
.self-start { align-self: flex-start; }
.self-center { align-self: center; }
.self-end { align-self: flex-end; }
.self-stretch { align-self: stretch; }

.grid-cols-1 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
.grid-cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid-cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.grid-cols-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.grid-cols-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.grid-cols-12 { grid-template-columns: repeat(12, minmax(0, 1fr)); }
.col-span-1 { grid-column: span 1 / span 1; }
.col-span-2 { grid-column: span 2 / span 2; }
.col-span-3 { grid-column: span 3 / span 3; }
.col-span-4 { grid-column: span 4 / span 4; }
.col-span-6 { grid-column: span 6 / span 6; }
.col-span-full { grid-column: 1 / -1; }

/* ── Gap (shared by flex + grid) ──────────────────────────────────── */
.gap-0 { gap: 0; }
.gap-1 { gap: 0.25rem; }
.gap-1\.5 { gap: 0.375rem; }
.gap-2 { gap: 0.5rem; }
.gap-3 { gap: 0.75rem; }
.gap-4 { gap: 1rem; }
.gap-5 { gap: 1.25rem; }
.gap-6 { gap: 1.5rem; }
.gap-8 { gap: 2rem; }
.gap-10 { gap: 2.5rem; }
.gap-12 { gap: 3rem; }
.gap-x-2 { column-gap: 0.5rem; }
.gap-x-4 { column-gap: 1rem; }
.gap-y-2 { row-gap: 0.5rem; }
.gap-y-4 { row-gap: 1rem; }

/* ── Padding / margin ─────────────────────────────────────────────── */
.p-0 { padding: 0; } .p-1 { padding: 0.25rem; } .p-1\.5 { padding: 0.375rem; }
.p-2 { padding: 0.5rem; } .p-3 { padding: 0.75rem; } .p-4 { padding: 1rem; }
.p-5 { padding: 1.25rem; } .p-6 { padding: 1.5rem; } .p-8 { padding: 2rem; }
.p-10 { padding: 2.5rem; } .p-12 { padding: 3rem; }

.px-0 { padding-left: 0; padding-right: 0; }
.px-1 { padding-left: 0.25rem; padding-right: 0.25rem; }
.px-2 { padding-left: 0.5rem; padding-right: 0.5rem; }
.px-3 { padding-left: 0.75rem; padding-right: 0.75rem; }
.px-4 { padding-left: 1rem; padding-right: 1rem; }
.px-5 { padding-left: 1.25rem; padding-right: 1.25rem; }
.px-6 { padding-left: 1.5rem; padding-right: 1.5rem; }
.px-8 { padding-left: 2rem; padding-right: 2rem; }
.px-10 { padding-left: 2.5rem; padding-right: 2.5rem; }

.py-0 { padding-top: 0; padding-bottom: 0; }
.py-1 { padding-top: 0.25rem; padding-bottom: 0.25rem; }
.py-1\.5 { padding-top: 0.375rem; padding-bottom: 0.375rem; }
.py-2 { padding-top: 0.5rem; padding-bottom: 0.5rem; }
.py-3 { padding-top: 0.75rem; padding-bottom: 0.75rem; }
.py-4 { padding-top: 1rem; padding-bottom: 1rem; }
.py-5 { padding-top: 1.25rem; padding-bottom: 1.25rem; }
.py-6 { padding-top: 1.5rem; padding-bottom: 1.5rem; }
.py-8 { padding-top: 2rem; padding-bottom: 2rem; }

.pt-2 { padding-top: 0.5rem; } .pt-4 { padding-top: 1rem; } .pt-6 { padding-top: 1.5rem; }
.pb-2 { padding-bottom: 0.5rem; } .pb-4 { padding-bottom: 1rem; } .pb-6 { padding-bottom: 1.5rem; }
.pl-3 { padding-left: 0.75rem; } .pl-4 { padding-left: 1rem; } .pr-3 { padding-right: 0.75rem; } .pr-4 { padding-right: 1rem; }

.m-0 { margin: 0; } .m-auto { margin: auto; }
.mx-0 { margin-left: 0; margin-right: 0; }
.mx-auto { margin-left: auto; margin-right: auto; }
.my-0 { margin-top: 0; margin-bottom: 0; }
.my-2 { margin-top: 0.5rem; margin-bottom: 0.5rem; }
.my-4 { margin-top: 1rem; margin-bottom: 1rem; }
.mt-1 { margin-top: 0.25rem; } .mt-2 { margin-top: 0.5rem; } .mt-3 { margin-top: 0.75rem; }
.mt-4 { margin-top: 1rem; } .mt-6 { margin-top: 1.5rem; } .mt-8 { margin-top: 2rem; }
.mb-1 { margin-bottom: 0.25rem; } .mb-2 { margin-bottom: 0.5rem; } .mb-3 { margin-bottom: 0.75rem; }
.mb-4 { margin-bottom: 1rem; } .mb-6 { margin-bottom: 1.5rem; } .mb-8 { margin-bottom: 2rem; }
.ml-1 { margin-left: 0.25rem; } .ml-2 { margin-left: 0.5rem; } .ml-3 { margin-left: 0.75rem; } .ml-auto { margin-left: auto; }
.mr-1 { margin-right: 0.25rem; } .mr-2 { margin-right: 0.5rem; } .mr-3 { margin-right: 0.75rem; } .mr-auto { margin-right: auto; }

/* ── Sizing ──────────────────────────────────────────────────────── */
.w-full { width: 100%; } .w-auto { width: auto; } .w-screen { width: 100vw; }
.w-2 { width: 0.5rem; } .w-3 { width: 0.75rem; } .w-4 { width: 1rem; } .w-5 { width: 1.25rem; }
.w-6 { width: 1.5rem; } .w-8 { width: 2rem; } .w-10 { width: 2.5rem; } .w-12 { width: 3rem; }
.w-16 { width: 4rem; } .w-20 { width: 5rem; } .w-24 { width: 6rem; } .w-32 { width: 8rem; }
.w-40 { width: 10rem; } .w-48 { width: 12rem; } .w-56 { width: 14rem; } .w-64 { width: 16rem; }
.w-72 { width: 18rem; } .w-80 { width: 20rem; } .w-96 { width: 24rem; }
.w-\[220px\] { width: 220px; }
.w-1\/2 { width: 50%; } .w-1\/3 { width: 33.333333%; } .w-2\/3 { width: 66.666667%; }
.w-1\/4 { width: 25%; } .w-3\/4 { width: 75%; }

.h-full { height: 100%; } .h-auto { height: auto; } .h-screen { height: 100vh; }
.h-2 { height: 0.5rem; } .h-3 { height: 0.75rem; } .h-4 { height: 1rem; } .h-5 { height: 1.25rem; }
.h-6 { height: 1.5rem; } .h-8 { height: 2rem; } .h-10 { height: 2.5rem; } .h-12 { height: 3rem; }
.h-14 { height: 3.5rem; } .h-16 { height: 4rem; } .h-20 { height: 5rem; } .h-24 { height: 6rem; }
.h-32 { height: 8rem; } .h-40 { height: 10rem; } .h-48 { height: 12rem; } .h-56 { height: 14rem; }
.h-64 { height: 16rem; } .h-72 { height: 18rem; } .h-80 { height: 20rem; } .h-96 { height: 24rem; }
.min-h-0 { min-height: 0; } .min-h-full { min-height: 100%; } .min-h-screen { min-height: 100vh; }
.min-w-0 { min-width: 0; }
.max-w-xs { max-width: 20rem; } .max-w-sm { max-width: 24rem; } .max-w-md { max-width: 28rem; }
.max-w-lg { max-width: 32rem; } .max-w-xl { max-width: 36rem; } .max-w-2xl { max-width: 42rem; }
.max-w-3xl { max-width: 48rem; } .max-w-4xl { max-width: 56rem; } .max-w-5xl { max-width: 64rem; }
.max-w-6xl { max-width: 72rem; } .max-w-7xl { max-width: 80rem; } .max-w-full { max-width: 100%; }

/* ── Typography ──────────────────────────────────────────────────── */
.font-sans { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
.font-mono { font-family: "JetBrains Mono", "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }

.text-\[11px\] { font-size: 11px; line-height: 1.4; }
.text-\[12px\] { font-size: 12px; line-height: 1.4; }
.text-\[13px\] { font-size: 13px; line-height: 1.5; }
.text-xs { font-size: 0.75rem; line-height: 1rem; }
.text-sm { font-size: 0.875rem; line-height: 1.25rem; }
.text-base { font-size: 1rem; line-height: 1.5rem; }
.text-lg { font-size: 1.125rem; line-height: 1.75rem; }
.text-xl { font-size: 1.25rem; line-height: 1.75rem; }
.text-2xl { font-size: 1.5rem; line-height: 2rem; }
.text-3xl { font-size: 1.875rem; line-height: 2.25rem; }
.text-4xl { font-size: 2.25rem; line-height: 2.5rem; }

.font-normal { font-weight: 400; }
.font-medium { font-weight: 500; }
.font-semibold { font-weight: 600; }
.font-bold { font-weight: 700; }

.leading-none { line-height: 1; } .leading-tight { line-height: 1.25; }
.leading-snug { line-height: 1.375; } .leading-normal { line-height: 1.5; }
.leading-relaxed { line-height: 1.625; } .leading-loose { line-height: 2; }

.tracking-tighter { letter-spacing: -0.05em; }
.tracking-tight { letter-spacing: -0.025em; }
.tracking-\[-0\.02em\] { letter-spacing: -0.02em; }
.tracking-normal { letter-spacing: 0; }
.tracking-wide { letter-spacing: 0.025em; }

.uppercase { text-transform: uppercase; }
.lowercase { text-transform: lowercase; }
.capitalize { text-transform: capitalize; }
.normal-case { text-transform: none; }

.text-left { text-align: left; } .text-center { text-align: center; } .text-right { text-align: right; }
.whitespace-nowrap { white-space: nowrap; }
.whitespace-pre-wrap { white-space: pre-wrap; }
.break-words { overflow-wrap: break-word; }
.truncate { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }

.tabular-nums { font-variant-numeric: tabular-nums; }

/* ── Text colour ─────────────────────────────────────────────────── */
.text-white { color: #ffffff; }
.text-black { color: #000000; }
.text-transparent { color: transparent; }

.text-slate-50 { color: #f8fafc; }
.text-slate-100 { color: #f1f5f9; }
.text-slate-200 { color: #e2e8f0; }
.text-slate-300 { color: #cbd5e1; }
.text-slate-400 { color: #94a3b8; }
.text-slate-500 { color: #64748b; }
.text-slate-600 { color: #475569; }
.text-slate-700 { color: #334155; }
.text-slate-800 { color: #1e293b; }
.text-slate-900 { color: #0f172a; }
.text-slate-950 { color: #020617; }

.text-violet-300 { color: #c4b5fd; }
.text-violet-400 { color: #a78bfa; }
.text-violet-500 { color: #8b5cf6; }
.text-violet-600 { color: #7c3aed; }

.text-cyan-300 { color: #67e8f9; }
.text-cyan-400 { color: #22d3ee; }

.text-emerald-400 { color: #34d399; }
.text-emerald-500 { color: #10b981; }
.text-amber-400 { color: #fbbf24; }
.text-amber-500 { color: #f59e0b; }
.text-rose-400 { color: #fb7185; }
.text-rose-500 { color: #f43f5e; }
.text-sky-400 { color: #38bdf8; }
.text-sky-500 { color: #0ea5e9; }

/* ── Background colour ───────────────────────────────────────────── */
.bg-transparent { background-color: transparent; }
.bg-white { background-color: #ffffff; }
.bg-black { background-color: #000000; }

.bg-slate-50 { background-color: #f8fafc; }
.bg-slate-100 { background-color: #f1f5f9; }
.bg-slate-200 { background-color: #e2e8f0; }
.bg-slate-700 { background-color: #334155; }
.bg-slate-800 { background-color: #1e293b; }
.bg-slate-900 { background-color: #0f172a; }
.bg-slate-950 { background-color: #020617; }

/* Slash-opacity variants used by design spec (§2). */
.bg-slate-800\/60 { background-color: rgb(30 41 59 / 0.6); }
.bg-slate-900\/60 { background-color: rgb(15 23 42 / 0.6); }
.bg-slate-900\/80 { background-color: rgb(15 23 42 / 0.8); }
.bg-slate-950\/40 { background-color: rgb(2 6 23 / 0.4); }
.bg-slate-950\/80 { background-color: rgb(2 6 23 / 0.8); }

.bg-violet-500 { background-color: #8b5cf6; }
.bg-violet-500\/10 { background-color: rgb(139 92 246 / 0.1); }
.bg-violet-500\/20 { background-color: rgb(139 92 246 / 0.2); }
.bg-violet-500\/30 { background-color: rgb(139 92 246 / 0.3); }
.bg-violet-600 { background-color: #7c3aed; }
.bg-cyan-400 { background-color: #22d3ee; }
.bg-cyan-400\/20 { background-color: rgb(34 211 238 / 0.2); }

.bg-emerald-500 { background-color: #10b981; }
.bg-emerald-500\/20 { background-color: rgb(16 185 129 / 0.2); }
.bg-emerald-500\/30 { background-color: rgb(16 185 129 / 0.3); }
.bg-amber-500 { background-color: #f59e0b; }
.bg-amber-500\/20 { background-color: rgb(245 158 11 / 0.2); }
.bg-rose-500 { background-color: #f43f5e; }
.bg-rose-500\/20 { background-color: rgb(244 63 94 / 0.2); }
.bg-rose-500\/30 { background-color: rgb(244 63 94 / 0.3); }
.bg-sky-400 { background-color: #38bdf8; }
.bg-sky-400\/20 { background-color: rgb(56 189 248 / 0.2); }

/* Brand gradient (primary CTAs, selected nav indicator, splash logo). */
.bg-gradient-to-r { background-image: linear-gradient(to right, var(--tw-gradient-from), var(--tw-gradient-to)); }
.bg-gradient-to-br { background-image: linear-gradient(to bottom right, var(--tw-gradient-from), var(--tw-gradient-to)); }
.from-violet-500 { --tw-gradient-from: #8b5cf6; }
.from-violet-600 { --tw-gradient-from: #7c3aed; }
.to-cyan-400 { --tw-gradient-to: #22d3ee; }
.to-cyan-300 { --tw-gradient-to: #67e8f9; }

/* ── Borders ─────────────────────────────────────────────────────── */
.border-0 { border-width: 0; }
.border { border-width: 1px; }
.border-2 { border-width: 2px; }
.border-t { border-top-width: 1px; }
.border-b { border-bottom-width: 1px; }
.border-l { border-left-width: 1px; }
.border-r { border-right-width: 1px; }
.border-l-2 { border-left-width: 2px; }

.border-slate-700 { border-color: #334155; }
.border-slate-800 { border-color: #1e293b; }
.border-slate-800\/60 { border-color: rgb(30 41 59 / 0.6); }
.border-slate-900 { border-color: #0f172a; }
.border-violet-500 { border-color: #8b5cf6; }
.border-violet-500\/40 { border-color: rgb(139 92 246 / 0.4); }
.border-cyan-400 { border-color: #22d3ee; }
.border-emerald-500 { border-color: #10b981; }
.border-amber-500 { border-color: #f59e0b; }
.border-rose-500 { border-color: #f43f5e; }
.border-transparent { border-color: transparent; }

.rounded-none { border-radius: 0; }
.rounded-sm { border-radius: 0.125rem; }
.rounded { border-radius: 0.25rem; }
.rounded-md { border-radius: 0.375rem; }
.rounded-lg { border-radius: 0.5rem; }
.rounded-xl { border-radius: 0.75rem; }
.rounded-2xl { border-radius: 1rem; }
.rounded-3xl { border-radius: 1.5rem; }
.rounded-full { border-radius: 9999px; }
.rounded-t-2xl { border-top-left-radius: 1rem; border-top-right-radius: 1rem; }
.rounded-b-2xl { border-bottom-left-radius: 1rem; border-bottom-right-radius: 1rem; }

/* ── Shadows ─────────────────────────────────────────────────────── */
.shadow-none { box-shadow: none; }
.shadow-sm { box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.3); }
.shadow { box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.4), 0 1px 2px -1px rgb(0 0 0 / 0.4); }
.shadow-md { box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.4), 0 2px 4px -2px rgb(0 0 0 / 0.4); }
.shadow-lg { box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.5), 0 4px 6px -4px rgb(0 0 0 / 0.5); }
.shadow-xl { box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.5), 0 8px 10px -6px rgb(0 0 0 / 0.5); }

/* ── Overflow ────────────────────────────────────────────────────── */
.overflow-auto { overflow: auto; }
.overflow-hidden { overflow: hidden; }
.overflow-visible { overflow: visible; }
.overflow-scroll { overflow: scroll; }
.overflow-x-auto { overflow-x: auto; }
.overflow-x-hidden { overflow-x: hidden; }
.overflow-y-auto { overflow-y: auto; }
.overflow-y-hidden { overflow-y: hidden; }

/* ── Cursor + user-select ────────────────────────────────────────── */
.cursor-auto { cursor: auto; }
.cursor-default { cursor: default; }
.cursor-pointer { cursor: pointer; }
.cursor-not-allowed { cursor: not-allowed; }
.cursor-text { cursor: text; }
.select-none { user-select: none; }
.select-text { user-select: text; }
.select-auto { user-select: auto; }

/* ── Pointer events ──────────────────────────────────────────────── */
.pointer-events-none { pointer-events: none; }
.pointer-events-auto { pointer-events: auto; }

/* ── Opacity ─────────────────────────────────────────────────────── */
.opacity-0 { opacity: 0; }
.opacity-25 { opacity: 0.25; }
.opacity-50 { opacity: 0.5; }
.opacity-60 { opacity: 0.6; }
.opacity-75 { opacity: 0.75; }
.opacity-90 { opacity: 0.9; }
.opacity-100 { opacity: 1; }

/* ── Transitions + animation ─────────────────────────────────────── */
.transition { transition-property: color, background-color, border-color, fill, stroke, opacity, box-shadow, transform; transition-timing-function: cubic-bezier(.4,0,.2,1); transition-duration: 150ms; }
.transition-colors { transition-property: color, background-color, border-color, fill, stroke; transition-timing-function: cubic-bezier(.4,0,.2,1); transition-duration: 150ms; }
.transition-opacity { transition-property: opacity; transition-timing-function: cubic-bezier(.4,0,.2,1); transition-duration: 150ms; }
.transition-transform { transition-property: transform; transition-timing-function: cubic-bezier(.4,0,.2,1); transition-duration: 150ms; }
.transition-all { transition-property: all; transition-timing-function: cubic-bezier(.4,0,.2,1); transition-duration: 150ms; }
.duration-75 { transition-duration: 75ms; }
.duration-100 { transition-duration: 100ms; }
.duration-150 { transition-duration: 150ms; }
.duration-200 { transition-duration: 200ms; }
.duration-300 { transition-duration: 300ms; }
.duration-500 { transition-duration: 500ms; }
.ease-linear { transition-timing-function: linear; }
.ease-in { transition-timing-function: cubic-bezier(.4,0,1,1); }
.ease-out { transition-timing-function: cubic-bezier(0,0,.2,1); }
.ease-in-out { transition-timing-function: cubic-bezier(.4,0,.2,1); }

@keyframes rade-spin { to { transform: rotate(360deg); } }
@keyframes rade-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
@keyframes rade-ping { 75%,100% { transform: scale(2); opacity: 0; } }
@keyframes rade-bounce { 0%,100% { transform: translateY(-25%); animation-timing-function: cubic-bezier(.8,0,1,1); } 50% { transform: translateY(0); animation-timing-function: cubic-bezier(0,0,.2,1); } }
.animate-spin { animation: rade-spin 1s linear infinite; }
.animate-pulse { animation: rade-pulse 2s cubic-bezier(.4,0,.6,1) infinite; }
.animate-ping { animation: rade-ping 1s cubic-bezier(0,0,.2,1) infinite; }
.animate-bounce { animation: rade-bounce 1s infinite; }

/* ── Transform ───────────────────────────────────────────────────── */
.transform { transform: translate(var(--tw-translate-x, 0), var(--tw-translate-y, 0)) rotate(var(--tw-rotate, 0)) skewX(var(--tw-skew-x, 0)) skewY(var(--tw-skew-y, 0)) scaleX(var(--tw-scale-x, 1)) scaleY(var(--tw-scale-y, 1)); }
.rotate-45 { --tw-rotate: 45deg; transform: rotate(45deg); }
.rotate-90 { --tw-rotate: 90deg; transform: rotate(90deg); }
.rotate-180 { --tw-rotate: 180deg; transform: rotate(180deg); }
.scale-95 { --tw-scale-x: .95; --tw-scale-y: .95; transform: scale(.95); }
.scale-100 { --tw-scale-x: 1; --tw-scale-y: 1; transform: scale(1); }
.scale-105 { --tw-scale-x: 1.05; --tw-scale-y: 1.05; transform: scale(1.05); }

/* ── Ring (focus + decoration) ───────────────────────────────────── */
.ring-0 { box-shadow: 0 0 0 0 transparent; }
.ring-1 { box-shadow: 0 0 0 1px rgb(139 92 246); }
.ring-2 { box-shadow: 0 0 0 2px rgb(139 92 246); }
.ring-violet-500 { box-shadow: 0 0 0 2px #8b5cf6; }
.ring-offset-2 { /* combined with ring-* via multi-shadow if needed — keep simple here */ }

/* ── Hover / focus (flat — keeps bundle readable) ────────────────── */
.hover\:bg-slate-800:hover { background-color: #1e293b; }
.hover\:bg-slate-800\/60:hover { background-color: rgb(30 41 59 / 0.6); }
.hover\:bg-violet-500\/10:hover { background-color: rgb(139 92 246 / 0.1); }
.hover\:bg-violet-600:hover { background-color: #7c3aed; }
.hover\:text-slate-100:hover { color: #f1f5f9; }
.hover\:text-slate-200:hover { color: #e2e8f0; }
.hover\:text-violet-400:hover { color: #a78bfa; }
.hover\:border-violet-500:hover { border-color: #8b5cf6; }
.hover\:border-slate-700:hover { border-color: #334155; }
.hover\:opacity-90:hover { opacity: 0.9; }
.focus\:outline-none:focus { outline: none; }

/* ── Misc ────────────────────────────────────────────────────────── */
.appearance-none { appearance: none; }
.list-none { list-style: none; }
.align-middle { vertical-align: middle; }
.backdrop-blur-sm { backdrop-filter: blur(4px); }

/* ─────────────────────────────────────────────────────────────────── */
/* COMPONENT CLASSES                                                    */
/* Mirrors @layer components in tailwind.input.css                      */
/* ─────────────────────────────────────────────────────────────────── */

.rade-card {
  border-radius: 1rem;
  background-color: #0f172a;
  border: 1px solid #1e293b;
  padding: 1.25rem;
}
.rade-card-compact {
  border-radius: 1rem;
  background-color: #0f172a;
  border: 1px solid #1e293b;
  padding: 1rem;
}
.rade-kpi-label {
  font-size: 11px;
  line-height: 1.4;
  text-transform: uppercase;
  letter-spacing: 0.025em;
  color: #64748b;
}
.rade-kpi-value {
  font-size: 1.5rem;
  line-height: 2rem;
  font-weight: 600;
  color: #f1f5f9;
  font-family: "JetBrains Mono", "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
.rade-nav-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 0.75rem;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  line-height: 1.25rem;
  color: #94a3b8;
  transition-property: color, background-color, border-color;
  transition-timing-function: cubic-bezier(.4,0,.2,1);
  transition-duration: 150ms;
}
.rade-nav-item:hover {
  background-color: #1e293b;
  color: #f1f5f9;
}
.rade-nav-item--active {
  background-color: #1e293b;
  color: #f1f5f9;
  border-left: 2px solid #8b5cf6;
}
.rade-brand-gradient {
  background-image: linear-gradient(to right, #8b5cf6, #22d3ee);
}
.rade-brand-gradient-text {
  background-image: linear-gradient(to right, #8b5cf6, #22d3ee);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

/* Skeleton shimmer — used by state_wrappers.Loading. */
.rade-skeleton-shimmer {
  position: relative;
  overflow: hidden;
  background: #0f172a;
  border-radius: 0.5rem;
}
.rade-skeleton-shimmer::after {
  content: "";
  position: absolute;
  inset: 0;
  transform: translateX(-100%);
  background-image: linear-gradient(
    90deg,
    rgba(148,163,184,0) 0%,
    rgba(148,163,184,0.08) 50%,
    rgba(148,163,184,0) 100%
  );
  animation: rade-shimmer 1.6s infinite;
}
@keyframes rade-shimmer {
  100% { transform: translateX(100%); }
}

/* AG Grid overrides — pair with className="ag-theme-alpine-dark rade-grid". */
.rade-grid {
  --ag-background-color: #0f172a;
  --ag-foreground-color: #e2e8f0;
  --ag-header-background-color: #0f172a;
  --ag-header-foreground-color: #94a3b8;
  --ag-odd-row-background-color: rgb(2 6 23 / 0.4);
  --ag-border-color: #1e293b;
  --ag-row-border-color: #1e293b;
  --ag-row-hover-color: rgb(30 41 59 / 0.6);
  --ag-selected-row-background-color: rgb(139 92 246 / 0.15);
  --ag-font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  --ag-font-size: 13px;
  border-radius: 1rem;
  overflow: hidden;
  border: 1px solid #1e293b;
}

/* ─────────────────────────────────────────────────────────────────── */
/* SPLASH PAGE                                                          */
/* Custom rules for layouts/splash.py — mirrors arbitrary-value         */
/* Tailwind utilities so the page renders without a Tailwind rebuild.   */
/* ─────────────────────────────────────────────────────────────────── */

.rade-splash-root {
  position: relative;
  min-height: 100vh;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.rade-splash-backdrop {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.rade-splash-glow-violet {
  position: absolute;
  top: -10rem;
  left: -10rem;
  width: 28rem;
  height: 28rem;
  border-radius: 9999px;
  background-color: rgb(139 92 246 / 0.18);
  filter: blur(64px);
}
.rade-splash-glow-cyan {
  position: absolute;
  bottom: -10rem;
  right: -10rem;
  width: 32rem;
  height: 32rem;
  border-radius: 9999px;
  background-color: rgb(34 211 238 / 0.12);
  filter: blur(80px);
}

.rade-splash-card {
  /* Layout only — the card is intentionally invisible so the hero
     above and the gradient backdrop read as a single scene.  Child
     elements (status strip, Select, Button) carry their own visual
     treatment; the outer card just sizes + spaces them. */
  position: relative;
  z-index: 10;
  width: 100%;
  max-width: 28rem;
  margin-left: 1rem;
  margin-right: 1rem;
  padding: 0;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 1rem;
  background-color: transparent;
  border: 0;
  box-shadow: none;
}

.rade-splash-brand-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  text-align: center;
}
.rade-splash-tagline {
  font-size: 0.875rem;
  color: #94a3b8;
  max-width: 24rem;
  line-height: 1.625;
}

.rade-splash-status-strip {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  background-color: rgb(15 23 42 / 0.6);
  border: 1px solid #1e293b;
}
.rade-splash-status-inner {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.rade-splash-status-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: #cbd5e1;
}
.rade-splash-api-url {
  font-size: 0.75rem;
  color: #64748b;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  max-width: 16rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.rade-splash-version-block {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}
.rade-splash-version-inner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.rade-section-label {
  font-size: 0.7rem;
  letter-spacing: 0.14em;
  font-weight: 600;
  color: #64748b;
  text-transform: uppercase;
}

.rade-splash-version-headline {
  font-size: 1.5rem;
  font-weight: 600;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  background-image: linear-gradient(to right, #8b5cf6, #22d3ee);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.rade-splash-cta-block {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.rade-splash-error-banner {
  display: none;
  font-size: 0.75rem;
  color: #fda4af;
  background-color: rgb(244 63 94 / 0.1);
  border: 1px solid rgb(244 63 94 / 0.4);
  border-radius: 0.375rem;
  padding: 0.5rem 0.75rem;
}
.rade-splash-error-banner--visible {
  display: block;
}

.rade-splash-footer {
  font-size: 0.7rem;
  color: #475569;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  letter-spacing: 0.05em;
}

.rade-status-dot {
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 9999px;
}
.rade-status-dot--ok      { background-color: #34d399; }
.rade-status-dot--err     { background-color: #fb7185; }
.rade-status-dot--booting {
  background-color: #64748b;
  animation: rade-status-pulse 1.5s ease-in-out infinite;
}
@keyframes rade-status-pulse {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.3; }
}

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

/* ── Cluster health heatmap (5-col grid of coloured cells) ────── */
/* Height-capped + scrollable so the parent card stays the same   */
/* height as the portfolio-chart card on the overview row, no     */
/* matter how many clusters the ensemble has.                     */

.rade-cluster-heatmap {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.5rem;
  padding: 0.25rem 0.4rem 0.25rem 0;   /* right padding clears scrollbar */
  /* Cap matches ChartContainer(height=280) on the overview chart. */
  max-height: 280px;
  overflow-y: auto;
  /* Firefox scrollbar */
  scrollbar-width: thin;
  scrollbar-color: #334155 transparent;
}

/* WebKit (Chrome / Safari / Edge) scrollbar — dark, slim, recessed */
.rade-cluster-heatmap::-webkit-scrollbar {
  width: 6px;
}
.rade-cluster-heatmap::-webkit-scrollbar-track {
  background: transparent;
}
.rade-cluster-heatmap::-webkit-scrollbar-thumb {
  background-color: #334155;          /* slate-700 */
  border-radius: 3px;
}
.rade-cluster-heatmap::-webkit-scrollbar-thumb:hover {
  background-color: #475569;          /* slate-600 */
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

/* ==================================================================
 * EVALUATION PAGE — global filter bar + sub-tabs  (Phase E.0)
 *
 * Visual anchor: `docs/platform_designs/rade_eval_portfolio.png`.
 *
 * Layout contract:
 *   .rade-evaluation                root, wraps filter bar + tabs + content
 *   .rade-filter-bar                two-row container; radius 10
 *     .rade-filter-row--one         collapsed state: toggle + chips + clear
 *     .rade-filter-row--two         expanded drawer with dropdowns (dmc.Collapse)
 *     .rade-filter-chip             dmc.Badge with close glyph
 *   .rade-evaluation-tabs           dmc.Tabs wrapper (pill variant)
 *   .rade-evaluation-content        sub-tab body slot
 *   .rade-evaluation-subtab--stub   placeholder card (Phase E.0 stubs)
 * ================================================================== */

/* ── Page container ─────────────────────────────────────────────── */

.rade-evaluation {
  display: flex;
  flex-direction: column;
  gap: 1rem;                            /* 16 px vertical rhythm */
  padding: 1.25rem 1.5rem;              /* match overview rade-page */
}

/* ── Filter bar root ────────────────────────────────────────────── */

.rade-filter-bar {
  background-color: #0f172a;            /* slate-900 */
  border: 1px solid #1e293b;            /* slate-800 */
  border-radius: 0.75rem;               /* 12 px */
  padding: 0.5rem 0.875rem;             /* tight top row; drawer adds its own */
}

.rade-filter-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.rade-filter-row--one {
  min-height: 36px;
}

.rade-filter-row--two {
  display: grid;
  grid-template-columns: repeat(5, minmax(160px, 1fr)) auto;
  gap: 0.75rem;
  align-items: flex-end;                /* reset button aligns with dropdown bottom */
  padding-top: 0.75rem;                 /* sit below the collapsed row */
  border-top: 1px solid #1e293b;
  margin-top: 0.5rem;
}

/* Keep the grid readable on narrow screens — wrap to 3 / 2 columns. */
@media (max-width: 1200px) {
  .rade-filter-row--two {
    grid-template-columns: repeat(3, minmax(160px, 1fr));
  }
}
@media (max-width: 800px) {
  .rade-filter-row--two {
    grid-template-columns: repeat(2, minmax(140px, 1fr));
  }
}

/* ── Row-1 pieces ───────────────────────────────────────────────── */

.rade-filter-left,
.rade-filter-right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  min-width: 0;
}

.rade-filter-count {
  font-size: 0.75rem;
  color: #94a3b8;                       /* slate-400 */
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.rade-filter-chips {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  flex-wrap: wrap;
  min-height: 1.5rem;                   /* keep row height stable when empty */
}

.rade-filter-chip {
  /* Self-contained pill — rendered identically by the Python initial
     build (render_filter_chips) and the clientside update callback in
     assets/js/evaluation.js, so neither path depends on dmc.Badge
     internals. */
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.125rem 0.25rem 0.125rem 0.5rem;
  background-color: rgba(167, 139, 250, 0.12);  /* violet-400 / 12 */
  color: #c4b5fd;                                /* violet-300 */
  border: 1px solid rgba(167, 139, 250, 0.2);
  border-radius: 0.375rem;
  font-size: 0.75rem;
  font-weight: 500;
  line-height: 1.25rem;
  white-space: nowrap;
}

.rade-filter-chip-label {
  /* The dimension + values text inside the chip.  Kept as its own
     class so the close button's hover state can use sibling CSS in
     the future without re-styling the whole chip. */
  font-variant-numeric: tabular-nums;
}

.rade-filter-chip-close {
  all: unset;                           /* strip button defaults */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 14px;
  height: 14px;
  margin-left: 2px;
  border-radius: 999px;
  font-size: 12px;
  line-height: 1;
  color: #cbd5e1;                       /* slate-300 */
  cursor: pointer;
  transition: background-color 120ms ease, color 120ms ease;
}
.rade-filter-chip-close:hover {
  background-color: rgba(148, 163, 184, 0.2);  /* slate-400 / 20 */
  color: #f8fafc;                       /* slate-50 */
}
.rade-filter-chip-close:focus-visible {
  outline: 2px solid #a78bfa;           /* violet-400 */
  outline-offset: 1px;
}

/* ── Row-2 dropdowns + reset ────────────────────────────────────── */

.rade-filter-dropdown {
  /* Nothing opinionated today — dmc MultiSelect already theme-aware in
     dark mode.  Kept as a named hook so future tweaks (width,
     padding, label colour) live in one place. */
}

.rade-filter-reset-wrap {
  display: flex;
  justify-content: flex-end;
  align-items: flex-end;
  padding-bottom: 1px;                  /* nudge to match MultiSelect input height */
}

/* ── Tabs row ───────────────────────────────────────────────────── */

.rade-evaluation-tabs {
  /* dmc.Tabs pills ship with their own styling; this class exists so
     a later phase can swap in a corporate accent colour without
     hunting through layouts. */
}

/* ── Content slot ──────────────────────────────────────────────── */

.rade-evaluation-content {
  min-height: 240px;                    /* keeps perceived height stable while stubs */
}

/* ── Sub-tab stub placeholder ──────────────────────────────────── */

.rade-evaluation-subtab--stub {
  /* The Empty card already supplies its own chrome (border, radius,
     padding); this wrapper just constrains width on ultra-wide
     screens so the message doesn't stretch across 3000 px. */
  max-width: 720px;
  margin: 1.5rem auto 0 auto;
}
```

---

## Appendix B — Page Template (v1.1)

A copy-paste-ready scaffold that captures the canonical page shape every
Rade Analytics page follows.  Lives in the source repo at
`docs/rade_analytics/page_template/` and pairs with `page_contract.md`.

The scaffold is three files:

| File | Purpose |
|---|---|
| `README.md` | 15-minute add-a-page workflow + helper cheat sheet + pre-flight checklist |
| `template_layout.py` | Canonical layout module with `mount_signal` Store, header band, KPI + chart row, `build_template(*, session)` |
| `template_cb.py` | Canonical callback module with capture/render split, mount-signal-triggered bootstrap (with URL deep-link + fresh-user override edges), pathname-gated render using `figure_with_fallback` |

Each file is reproduced verbatim below for easy copy-paste into a
work-env template folder.

### Reusing in a separate work-env template folder — caveats

The user-facing concerns when replicating this scaffold into a separate
templates folder (outside the source repo) are all manageable, but worth
calling out so they don't bite later:

| Concern | Severity | Mitigation |
|---|---|---|
| **Drift between template and source patterns.**  The template encodes contract rules as of v1.1.  When `page_contract.md` evolves (new rules, refactored helpers, new component primitives), the work-env copy goes stale.  New pages built from a stale template will silently violate current rules. | Medium | Version-tag the template directory (e.g. `page_template_v1_1/`), and re-sync from the source repo when the contract bumps.  The contract's change-log lists each version's diff so you know what to update. |
| **Relative imports lint as unresolved outside the package.**  The two `.py` files use `from ..data.result_helpers import ...` etc.  These resolve correctly only when the file lives inside `src/ui/apps/rade_analytics/`.  If you put the templates in a generic `templates/` folder (outside the package), your IDE will flag every relative import as red. | Low (cosmetic) | The files are documented "DO NOT IMPORT" — the lint warnings have no runtime impact.  Either live with them, or add `# noqa: F401` / `# pyright: ignore` annotations during the copy-paste phase. |
| **`page_contract.md` cross-links break.**  The `README.md` links to `../page_contract.md`.  If your work-env folder doesn't ship that file at the relative path, the link is dead. | Low | Either copy `page_contract.md` next to the templates, or rewrite the link in your copy.  The README's helper-cheat-sheet section still functions standalone. |
| **Per-org component drift.**  If your work-env app uses different component primitives (e.g. a different KPI card, no `ChartContainer`), the template needs adapting.  The same applies to session field names (`session.evaluation.deep_dive_cluster_id` is specific to this app's Session schema). | Low-Medium | Template imports are intentionally narrow (3 components, 1 helper).  Search-replace those four imports to match your work-env's primitives.  The structural pattern (mount_signal, capture/render split, pathname gate) is component-agnostic. |
| **`figure_with_fallback` / `component_with_fallback` not in the work-env.**  These helpers live in `data/result_helpers.py` in the source repo. | Low | Either copy that module across too (it's ~80 LOC, no external deps beyond `pandas` + `plotly` + `state_wrappers`), or strip the helper imports and inline the tri-state branches manually.  The README's "When the helper doesn't fit" section covers the inline form. |

**Net assessment:** putting these in a reusable templates folder is a
good idea.  Every concern above is mechanical and one-time at the copy
boundary.  The structural payoff — every new page starts contract-
compliant, in 15 minutes — is the same as adopting the source-repo
scaffold.  Just version-tag the folder and re-sync on contract bumps.

---

### B.1 — `page_template/README.md`

Save as `<template-folder>/README.md`.

````markdown
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
- [ ] `mount_signal` Store is in the layout root and is the trigger Input
      for the bootstrap callback.
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
- [ ] `prevent_initial_call=False` only on the bootstrap callback +
      pathname-gated render callbacks; everything else `True`.
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
````

---

### B.2 — `page_template/template_layout.py`

Save as `<template-folder>/template_layout.py`.

```python
"""TEMPLATE — Layout module skeleton for a new Rade Analytics page.

DO NOT IMPORT THIS FILE.  Copy it to
``src/ui/apps/rade_analytics/layouts/<your_page>.py`` and replace every
``# TODO:`` marker.  See ``docs/rade_analytics/page_template/README.md``
for the 15-minute add-a-page workflow.

What this template demonstrates
-------------------------------
* **Pure layout** (Page Contract §3 Rule L1) — ``build_template(*, session)``
  takes a :class:`Session` and bakes initial input values into the layout
  at build time.  No hydration callback after mount.
* **Mount tripwire** — a ``dcc.Store(id=mount_signal, data=True)`` mounted
  at the layout root.  The bootstrap callback in ``template_cb.py``
  triggers off this Store's ``data`` Input so it fires exactly once per
  fresh mount of the page (Page Contract §3 Rule L4).
* **Stable id contract** — every component id lives in :data:`TEMPLATE_IDS`
  so callbacks never hardcode strings (Page Contract §3 Rule L3).
* **Tailwind / dmc layout primitives** — header band → KPI card → chart
  body, matching the rest of the app.

Anatomy
-------
::

    Row 0 · mount tripwire (invisible)
    Row 1 · Header band (context picker · open-related-page link)
    Row 2 · Single KPI card  ·  Single time-series chart
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ...components.chart_container import ChartContainer
from ...components.kpi_card import KpiCard
from ...data.session import Session


# TODO: Rename TEMPLATE_IDS → <YOUR_PAGE>_IDS, and prefix every id value
# with "your-page-" so the Dash dev-tools error messages tell you which
# page they came from.
TEMPLATE_IDS: Dict[str, str] = {
    "root":                "template-root",

    # Row 1 — Header band
    "context_select":      "template-context-select",
    "open_related_btn":    "template-open-related-btn",

    # Row 2 — KPI card
    "kpi_value":           "template-kpi-value",
    "kpi_card":            "template-kpi-card",

    # Row 2 — Chart
    "main_chart":          "template-main-chart",

    # Mount tripwire — Page Contract §3 Rule L4.  A memory-store seeded
    # with ``data=True`` at build time; the bootstrap callback uses it
    # as its trigger Input so it fires *exactly once* per fresh mount,
    # *after* the parent has finished writing the new content tree.
    # Pathname-as-Input would race with that write; top-level-store
    # would only fire on cross-top-level navigation.
    "mount_signal":        "template-mount-signal",

    # Optional ephemeral stores (uncomment if your page needs them)
    # "store_<thing>":     "template-<thing>-store",
}


# ─────────────────────────────────────────────────────────────────────
# Row 1 — Header band
# ─────────────────────────────────────────────────────────────────────


def _header_band(
    *,
    initial_context_id: Optional[str] = None,
) -> html.Div:
    """Sticky header — context picker + cross-link.

    Parameters
    ----------
    initial_context_id
        Seed for the :class:`dmc.Select`'s ``value`` prop.  Sourced
        from ``session`` at build time (Page Contract §3 Rule L1) —
        the bootstrap callback will populate the ``data`` (option
        list) prop after a backend lookup; the seeded ``value`` then
        reads against that fresh list.  When ``None``, the picker
        shows its placeholder until the user / bootstrap picks one.
    """
    context_picker = html.Div(
        className="flex flex-col gap-1 min-w-[220px]",
        children=[
            html.Span(
                # TODO: replace with your page's context noun (e.g. "Cluster", "Desk", "Run").
                "Context",
                className="text-[11px] uppercase tracking-wider text-slate-400",
            ),
            dmc.Select(
                id=TEMPLATE_IDS["context_select"],
                # ``data`` is intentionally empty here — the bootstrap
                # callback populates it after a backend lookup.
                # Seeding ``value`` from session means the picker
                # remembers the user's choice across page navigation.
                data=[],
                value=initial_context_id,
                # TODO: replace with a context-appropriate placeholder.
                placeholder="Select…",
                searchable=True,
                clearable=False,
                size="sm",
            ),
        ],
    )

    open_related_btn = dmc.Button(
        # TODO: rename + retarget the cross-link, or delete this if your
        # page doesn't have a partner page.
        "Related page",
        id=TEMPLATE_IDS["open_related_btn"],
        variant="light",
        color="violet",
        size="sm",
        leftSection=DashIconify(icon="tabler:share-2", width=16),
    )

    return html.Div(
        className="rade-card flex flex-col gap-3 sticky top-0 z-10",
        children=[
            html.Div(
                className="flex items-end gap-4 flex-wrap",
                children=[
                    context_picker,
                    html.Div(className="flex-1"),  # spacer
                    open_related_btn,
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — KPI + Chart
# ─────────────────────────────────────────────────────────────────────


def _row_kpi_chart() -> html.Div:
    """Two-column row — KPI card on the left, time-series chart on the right.

    The KPI card and chart both render placeholder content (an em-dash
    value, an empty figure) so the page never reflows on initial paint;
    render callbacks replace the placeholders with real values once the
    backend fetch returns.
    """
    return html.Div(
        # 2/5 : 3/5 split on wide screens; collapses to a single stack
        # on narrow viewports.
        className="grid grid-cols-1 lg:grid-cols-5 gap-3 items-stretch",
        children=[
            html.Div(
                className="lg:col-span-2 flex flex-col gap-3",
                children=[
                    KpiCard(
                        # TODO: pick a metric label that matches what
                        # the render callback computes (e.g. "MAE",
                        # "Trade count", "Coverage").
                        label="Headline metric",
                        value="—",
                        card_id=TEMPLATE_IDS["kpi_card"],
                        value_id=TEMPLATE_IDS["kpi_value"],
                        icon="tabler:chart-dots",
                    ),
                ],
            ),
            html.Div(
                className="lg:col-span-3 flex flex-col gap-3",
                children=[
                    ChartContainer(
                        # TODO: pick a title that matches the chart.
                        title="Time-series",
                        chart_id=TEMPLATE_IDS["main_chart"],
                    ),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Public entrypoint — build_<your_page>
# ─────────────────────────────────────────────────────────────────────


def build_template(*, session: Session) -> html.Div:
    """Build the page layout, seeded from :class:`Session`.

    Parameters
    ----------
    session
        The active session.  Drives initial values for every input on
        the page — e.g. ``session.evaluation.<your_page>_context_id``
        for the context picker.  Page Contract §3 Rule L1 mandates
        that every input that *can* read from session, *does* —
        eliminating the need for a hydration callback after mount.

    Notes
    -----
    The layout function is **pure** — same session in, same DOM out.
    All side effects (backend fetches, option-list population) happen
    in the bootstrap callback in ``<your_page>_cb._register_render``.
    """
    # TODO: replace with the actual session field your page reads.
    # If the field doesn't exist yet, add it to
    # ``data/session.py::EvaluationFilters`` (or the session root
    # for non-Evaluation pages) following the existing pattern.
    initial_context_id: Optional[str] = None  # session.evaluation.<your_field>

    return html.Div(
        id=TEMPLATE_IDS["root"],
        className="flex flex-col gap-4 p-4",
        children=[
            # Mount tripwire — Page Contract §3 Rule L4.
            dcc.Store(
                id=TEMPLATE_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            _header_band(initial_context_id=initial_context_id),
            _row_kpi_chart(),
        ],
    )


__all__ = [
    "TEMPLATE_IDS",
    "build_template",
]
```

---

### B.3 — `page_template/template_cb.py`

Save as `<template-folder>/template_cb.py`.

```python
"""TEMPLATE — Callback module skeleton for a new Rade Analytics page.

DO NOT IMPORT THIS FILE.  Copy it to
``src/ui/apps/rade_analytics/callbacks/<your_page>_cb.py`` and replace
every ``# TODO:`` marker.  Pair with ``template_layout.py`` from the
same folder.  See ``docs/rade_analytics/page_template/README.md`` for
the 15-minute add-a-page workflow.

Page Contract structure
-----------------------
The public surface is a single :func:`register` that delegates to two
section helpers, matching Page Contract §2 (capture / render split):

* :func:`_register_capture` — user-input gestures → :class:`Session`
  writes (no UI side-effects, no backend access).
* :func:`_register_render`  — state → DOM updates (no Session writes,
  except for the narrow ``_register_bootstrap`` capture-edge that
  handles URL deep-links + fresh-user defaults).

Capture (1 callback in this template)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``_sync_context`` — context-picker ``value`` → session (the field
  your page persists, e.g. ``deep_dive_cluster_id``).

Render (2 callbacks in this template)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``_bootstrap`` — mount_signal-triggered fetch of the context picker
  ``Select.data`` option list (version-keyed metadata that genuinely
  needs a backend round-trip).  Also handles two narrow override paths
  in the same return tuple — URL deep-link, fresh-user default — so
  the bootstrap is the single capture-edge in the render section.
* ``_render_main`` — session → KPI value + main chart figure.

Initial UI state (no value-side hydration)
------------------------------------------
The context picker's ``value`` prop is seeded from session at layout
build time (Page Contract §3 Rule L1) — see ``template_layout.py``'s
``build_template(*, session)``.  This module never writes
``context_select.value`` except in the bootstrap's two narrow override
cases.  Eliminating value-side hydration means the page paints the
user's previously-chosen state on first frame rather than after a
callback round-trip.

Why pathname-gating
-------------------
Page Contract §4 Rule C2 — every render callback gates on
``pathname == _<PAGE>_PATH`` to avoid wasted compute on cross-page
re-renders.  Cheap, idempotent, prevents stale-DOM warnings.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from dash import Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

# TODO: replace template_layout with the layout module you cp'd from
# the template (the import path will be ``..layouts.<your_page>``).
from ..layouts.template_layout import TEMPLATE_IDS
from ..layouts.shell import SHELL_IDS

from ..data.result_helpers import figure_with_fallback
from ..data.session import Session

# TODO: import the figure helpers your render callback needs.  Examples:
#   from ..figures import portfolio_pnl, error_over_time
# from ..figures import empty_figure  # only if you need it outside the helpers

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants — page route + initial values
# ─────────────────────────────────────────────────────────────────────

# TODO: set this to the route the router maps to your page (e.g.
# ``/evaluation/cross-cluster``, ``/data-quality``).  Every render
# callback gates on ``pathname == _TEMPLATE_PATH``.
_TEMPLATE_PATH = "/template"

# TODO: replace with your page's domain-appropriate empty placeholder
# (e.g. ``"—"`` for KPI values, ``[]`` for AgGrid rowData).
_PLACEHOLDER = "—"


# ═════════════════════════════════════════════════════════════════════
# Public surface
# ═════════════════════════════════════════════════════════════════════


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every callback for this page to ``app``.

    The two section helpers are the only top-level symbols a reader
    should need to scan to understand the page's wiring.

    Parameters
    ----------
    app
        The Dash app returned by :func:`rade_analytics.app.create_app`.
    backend
        Shared :class:`RadeBackend` — all data fetches go through here.
    """
    _register_capture(app)
    _register_render(app, backend)


# ─────────────────────────────────────────────────────────────────────
# Section dispatchers — capture / render split (Page Contract §2)
# ─────────────────────────────────────────────────────────────────────


def _register_capture(app: "Dash") -> None:
    """Attach capture-side callbacks: input gestures → session writes.

    Capture callbacks are forbidden from doing UI rendering and from
    touching ``backend`` (Page Contract §4 Rule C1).  They write only
    to the session-store; render callbacks pick up the writes via the
    session-store ``Input``.
    """
    _register_sync_context(app)


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Attach render-side callbacks: state → DOM, no session writes.

    Render callbacks consume URL + session-store as Inputs / States,
    do backend lookups via ``backend``, and emit values + figures into
    the page's components.  They never write to the session-store
    (which would cascade into other pages' render callbacks).

    The single exception is the bootstrap callback's two narrow
    override paths — URL deep-link, fresh-user default — which write
    both the ``Select.value`` and the session-store in the same return
    tuple to avoid a second capture round-trip.
    """
    _register_bootstrap(app, backend)
    _register_render_main(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. Capture — context picker → session
# ═════════════════════════════════════════════════════════════════════


def _register_sync_context(app: "Dash") -> None:
    """Persist the context picker's ``value`` into session."""

    @app.callback(
        Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
        Input(TEMPLATE_IDS["context_select"], "value"),
        State(SHELL_IDS["session_store"], "data"),
        # Page Contract §4 Rule C5 — capture callbacks default to
        # prevent_initial_call=True; we only want this to fire on a
        # genuine user pick, not on the fresh page paint.
        prevent_initial_call=True,
    )
    def _sync(
        new_context_id: Optional[str],
        session_data:   Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        # TODO: replace with the field on EvaluationFilters / Session
        # your page persists.  Example:
        #   session.evaluation.deep_dive_cluster_id = new_context_id
        if new_context_id is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        # session.evaluation.<your_field> = new_context_id  # TODO
        return session.to_store()


# ═════════════════════════════════════════════════════════════════════
# 2. Render — bootstrap (mount_signal → option list + override edges)
# ═════════════════════════════════════════════════════════════════════


def _register_bootstrap(app: "Dash", backend: "RadeBackend") -> None:
    """Fetch the context picker's option list once per fresh mount.

    The mount tripwire pattern (Page Contract §3 Rule L4) — this
    callback is the *only* render callback in the module that's
    allowed to write to the session-store, and only along two narrow
    edges:

    * **URL deep-link** — when ``?ctx=<id>`` differs from
      ``session.<field>``, copy URL → session + ``Select.value``.
    * **Fresh-user default** — when neither URL nor session has a
      value, pick the first option from the fetched list, write it to
      both ``Select.value`` and session.

    Both edges are handled in the same return tuple so the bootstrap
    is the single capture-edge in the render section; we avoid a
    second round-trip for these initial-state cases.
    """

    @app.callback(
        Output(TEMPLATE_IDS["context_select"],   "data"),
        Output(TEMPLATE_IDS["context_select"],   "value"),
        Output(SHELL_IDS["session_store"],       "data", allow_duplicate=True),
        Input(TEMPLATE_IDS["mount_signal"],      "data"),
        State(SHELL_IDS["url"],                  "search"),
        State(SHELL_IDS["session_store"],        "data"),
        # Page Contract §4 Rule C5 — explicit ``initial_duplicate``
        # because we share an Output with the capture callback.  The
        # mount_signal Input only fires once per fresh mount, so we
        # pay this cost exactly when we need the bootstrap to run.
        prevent_initial_call="initial_duplicate",
    )
    def _bootstrap(
        _trigger:     Any,
        url_search:   Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], Any, Any]:
        # TODO: backend lookup.  The fetch should be cheap (cache hit
        # after the first call) and version-keyed if your option list
        # depends on the active ensemble version.
        #
        # Example:
        #   res = backend.context_options()
        #   if not res.ok or not res.data:
        #       return [], no_update, no_update
        #   options = [{"value": c.id, "label": c.name} for c in res.data]
        options: List[Dict[str, str]] = []  # TODO

        session = Session.from_store(session_data)
        # TODO: replace with the field your page persists.
        session_value: Optional[str] = None  # session.evaluation.<your_field>

        # Override-edge 1 — URL deep-link.  TODO: wire up if your page
        # supports ``?ctx=<id>`` style deep-links.  Otherwise drop.
        url_value: Optional[str] = None
        # if url_search:
        #     parsed = parse_qs(url_search.lstrip("?"))
        #     url_value = (parsed.get("ctx") or [None])[0]

        if url_value and url_value != session_value:
            # session.evaluation.<your_field> = url_value  # TODO
            return options, url_value, session.to_store()

        # Override-edge 2 — fresh-user default.
        if not session_value and options:
            default_value = options[0]["value"]
            # session.evaluation.<your_field> = default_value  # TODO
            return options, default_value, session.to_store()

        # Steady state — just publish the option list, leave value +
        # session alone.
        return options, no_update, no_update


# ═════════════════════════════════════════════════════════════════════
# 3. Render — main KPI + chart
# ═════════════════════════════════════════════════════════════════════


def _register_render_main(app: "Dash", backend: "RadeBackend") -> None:
    """Render the headline KPI value + the time-series chart.

    Demonstrates:

    * **Pathname gate** — Page Contract §4 Rule C2.
    * **Tri-state via ``figure_with_fallback``** — Page Contract §8.
    * **uirevision keyed on the data domain** — Page Contract §6.
    """

    @app.callback(
        Output(TEMPLATE_IDS["kpi_value"],   "children"),
        Output(TEMPLATE_IDS["main_chart"],  "figure"),
        Input(SHELL_IDS["url"],             "pathname"),
        Input(SHELL_IDS["session_store"],   "data"),
        # Page Contract §4 Rule C5 — explicit opt-in.  Direct entry on
        # the page route (URL share, refresh) must paint the KPI +
        # chart on first frame; the layout-time render only has the
        # ``"—"`` placeholder + the chart container's empty figure.
        # The pathname guard in the body keeps non-route hits cheap.
        prevent_initial_call=False,
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any]:
        if pathname != _TEMPLATE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        split = session.split

        # TODO: read the page's persisted context id from session.
        context_id: Optional[str] = None  # session.evaluation.<your_field>

        if not context_id:
            # Pre-fetch guard — no BackendResult to classify yet.
            from ..figures import empty_figure
            return _PLACEHOLDER, empty_figure(
                "Pick a context above to see this page's metrics."
            )

        # TODO: backend lookup.  Returns a BackendResult[<DataFrame or DTO>].
        #   res = backend.<your_method>(split, context_id=context_id)
        from ..data.backend import BackendResult
        import pandas as pd  # TODO: drop if you don't need the placeholder
        res: BackendResult[pd.DataFrame] = BackendResult.success(pd.DataFrame())

        # KPI value — demonstrates the manual tri-state branch when
        # the helper doesn't fit (multiple Outputs of different types).
        if not res.ok:
            kpi_text = _PLACEHOLDER
        elif res.data is None or res.data.empty:
            kpi_text = _PLACEHOLDER
        else:
            # TODO: compute the headline metric from res.data.
            kpi_text = "0.0000"

        # Figure — uses figure_with_fallback to collapse tri-state.
        # TODO: replace the on_ok lambda with your figure helper, e.g.
        #   on_ok=lambda df: portfolio_pnl(df, uirevision_key=split),
        ui_key = f"{split}::{context_id}"
        from ..figures import empty_figure  # TODO: drop after you wire on_ok
        fig = figure_with_fallback(
            res,
            on_ok=lambda df: empty_figure(  # TODO: replace
                "TODO: render your time-series figure here."
            ).update_layout(uirevision=ui_key) or empty_figure("TODO"),
            empty_msg="No data for the active selection.",
        )

        return kpi_text, fig
```
