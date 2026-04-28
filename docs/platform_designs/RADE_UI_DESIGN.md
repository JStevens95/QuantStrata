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
