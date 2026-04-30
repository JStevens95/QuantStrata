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

## Appendix A — Governance page (Phase E.x)

Single appendix (replaces all prior appendix blocks). Wires the
`/governance` route end-to-end against `rade_governance.png`:

* New PRISM API endpoint `GET /prism/v1/governance/registry` —
  cross-version walk of the ensemble registry, returning one row per
  registered version.
* `RadeApiClient` + `RadeBackend` plumbing — typed wrapper, cache
  binding (60 s TTL so registry mutations show up within a minute) and
  a flattened `governance_registry_df()` accessor that stashes
  `active_version` on `df.attrs`.
* Dash layout (`layouts/governance.py`) — header band (4 KPIs +
  status filter + Promote button), Model Registry AG Grid, Lineage
  Timeline + Approvals row, static Audit Log grid.
* Dash callbacks (`callbacks/governance_cb.py`) — two render
  callbacks both gated on `mount_signal` + a pathname guard
  (Page Contract §4 Rule C7); status filter is **ephemeral V1** (not
  persisted in `Session`).
* CSS additions (status pills, lineage rows, page title, mono cell).
* Router (`router.py`) + callback dispatcher (`callbacks/__init__.py`)
  swaps in the live `build_governance` and registers `governance_cb`.

### How to apply

There are eleven targets in this appendix.  Five are **new files** —
copy the contents into the path called out in the section header.  Six
are **patches to existing files** — apply the targeted find/replace
shown in each section.

| § | Path | Action |
|---|---|---|
| A.1  | `src/rade_ml_pt/ensemble/api/models/governance.py`         | NEW |
| A.2  | `src/rade_ml_pt/ensemble/api/services/governance.py`       | NEW |
| A.3  | `src/rade_ml_pt/ensemble/api/routers/governance.py`        | NEW |
| A.4  | `src/rade_ml_pt/ensemble/api/app.py`                       | PATCH |
| A.5  | `src/rade_ml_pt/ensemble/api/client.py`                    | PATCH |
| A.6  | `src/ui/apps/rade_analytics/data/backend.py`               | PATCH |
| A.7  | `src/ui/apps/rade_analytics/layouts/governance.py`         | NEW |
| A.8  | `src/ui/apps/rade_analytics/callbacks/governance_cb.py`    | NEW |
| A.9  | `src/ui/apps/rade_analytics/router.py`                     | PATCH |
| A.10 | `src/ui/apps/rade_analytics/callbacks/__init__.py`         | PATCH |
| A.11 | `src/ui/apps/rade_analytics/assets/rade.css`               | APPEND |

No layout / CSS changes to other pages, no new third-party dependency,
no `Session` schema bump.  See §A.12 for the verification checklist.

---

### A.1 — `src/rade_ml_pt/ensemble/api/models/governance.py` (NEW)

```python
"""Pydantic schemas for the ``/prism/v1/governance/*`` endpoints.

Governance is a *cross-version* concern — every other PRISM endpoint
serves the active ensemble's evaluation bundle, but the governance
page needs to enumerate every version registered under the
``EnsembleRegistry`` so reviewers can see the full release history.

The shape of one row is deliberately wide because the UI Registry
table renders nine columns (status, version, created_by, created_at,
promoted_at, commit_sha, mae_test, n_clusters, n_trades) and we'd
rather pay the JSON inflation than do nine separate lookups in the
front-end.

Audit-log + sign-offs are intentionally *not* on the wire yet — the
V1 of the page seeds those panels from static placeholders so we can
ship the registry view immediately.  Re-introducing them is a Stage-2
add: extend this file with two more response types and add matching
endpoints; the UI is already set up to swallow the new fields.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GovernanceRegistryRow(BaseModel):
    """One row of the model registry — one ensemble version."""

    version: str = Field(
        ..., description="Concrete version directory name (e.g. ``ens_20260417_142233_abc123``).",
    )
    tags: List[str] = Field(
        default_factory=list,
        description=(
            "All tags pointing at this version in ``index.json`` (e.g. "
            "``['production']``, ``['latest']``, or both)."
        ),
    )
    status: str = Field(
        ...,
        description=(
            "Lifecycle bucket derived from the tag set: "
            "``'production'`` (carries the ``production`` tag), "
            "``'staging'`` (``staging`` tag), "
            "``'candidate'`` (the active / ``latest`` version when no "
            "production / staging tag has been set yet), or "
            "``'archived'`` (no tags pointing at it)."
        ),
    )
    created_at: str = Field(
        ...,
        description=(
            "ISO-formatted UTC timestamp.  Parsed from the version directory "
            "name when possible (versions follow the ``ens_YYYYMMDD_HHMMSS_*`` "
            "convention) and falls back to the directory's ``mtime``."
        ),
    )
    promoted_at: Optional[str] = Field(
        None,
        description=(
            "When the version was last tagged ``production`` — taken from the "
            "mtime of ``index.json`` for now (V1 placeholder).  Re-promotion "
            "events will get a proper audit trail in Stage 2."
        ),
    )
    created_by: str = Field(
        ...,
        description=(
            "Author / pipeline identity.  Currently a static seed (``pipeline-bot``) "
            "until the registry stamps a real author into version metadata."
        ),
    )
    commit_sha: str = Field(
        ...,
        description=(
            "Short git SHA proxy — the trailing hash component of the version "
            "directory name (the registry already embeds an md5-derived 6-char "
            "tag).  Replaced with the real source-control SHA in Stage 2."
        ),
    )
    n_members: int = Field(
        ...,
        description="Number of cluster members in this version's ``ensemble_config.json``.",
    )
    n_trades: int = Field(
        ...,
        description=(
            "Total trade count across all clusters' ``cluster_mapping`` entries — "
            "the population the ensemble was trained on."
        ),
    )
    aggregation: str = Field(
        ...,
        description="Aggregation strategy from ``ensemble_config.json`` (e.g. ``concat``, ``stacking``).",
    )
    has_evaluation: bool = Field(
        ...,
        description=(
            "Whether ``{artifacts_dir}/ensemble/{version}/evaluation/`` exists — "
            "tells the UI whether the row's *Open* button should land on the "
            "Evaluation tab or display 'Eval pending'."
        ),
    )
    mae_test: Optional[float] = Field(
        None,
        description=(
            "Test-split MAE if the evaluation bundle was published.  ``None`` "
            "for versions that were registered but never evaluated, or "
            "evaluations that didn't include a ``test`` split."
        ),
    )
    rmse_test: Optional[float] = Field(
        None,
        description="Test-split RMSE; same caveats as :attr:`mae_test`.",
    )
    coverage_test: Optional[float] = Field(
        None,
        description=(
            "Fraction of test trades the ensemble produced predictions for "
            "(0–1, expected to be ~1.0 for a healthy run).  ``None`` when "
            "evaluation hasn't run."
        ),
    )


class GovernanceRegistryResponse(BaseModel):
    """Collection response for ``GET /prism/v1/governance/registry``."""

    active_version: str = Field(
        ...,
        description=(
            "The currently-served version — the row whose ``version`` matches "
            "this gets a *Selected* outline in the UI table."
        ),
    )
    rows: List[GovernanceRegistryRow] = Field(
        ...,
        description=(
            "Every registered version, sorted newest-first by "
            ":attr:`GovernanceRegistryRow.created_at`."
        ),
    )
```

---

### A.2 — `src/rade_ml_pt/ensemble/api/services/governance.py` (NEW)

```python
"""Governance service — walks the cross-version ensemble registry.

Sits *next to* :mod:`reader` rather than on it because the
:class:`~src.rade_ml_pt.ensemble.api.services.reader.ArtifactReader` is
scoped to a single ensemble version (the active one), whereas
governance needs every version that's ever been registered.

What we read
------------
* ``{registry_dir}/ensemble/index.json`` — the tag → version map.  We
  invert it so each version carries the list of tags pointing at it.
* ``{registry_dir}/ensemble/{version}/ensemble_config.json`` — for
  ``n_members``, ``n_trades`` and ``aggregation``.
* The version directory's ``mtime`` — fallback for ``created_at`` when
  the version name doesn't follow the ``ens_YYYYMMDD_HHMMSS_*`` convention.
* ``{artifacts_dir}/ensemble/{version}/evaluation/ensemble_metrics.parquet`` —
  for ``mae_test`` / ``rmse_test`` / ``coverage_test``.  Optional —
  versions registered but not yet evaluated leave these as ``None``.

What we *don't* read
--------------------
* No git lookups, no author resolution — V1 seeds those columns with
  static placeholders and we'll add a real source-of-truth in Stage 2
  (probably stamped at registration time by the eval pipeline).
* No audit-log persistence — the UI seeds that panel with mocked
  events for the same Stage-2 reason.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.rade_ml_pt.ensemble.api.models.governance import (
    GovernanceRegistryResponse,
    GovernanceRegistryRow,
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────

_INDEX_FILENAME = "index.json"
_CONFIG_FILENAME = "ensemble_config.json"
_EVAL_METRICS_FILENAME = "ensemble_metrics.parquet"

# Until the registry stamps real authors, every row gets the same
# placeholder — keeps the table column populated without lying about
# provenance.  Replaced with a real lookup in Stage 2.
_DEFAULT_AUTHOR = "pipeline-bot"

# Versions follow ``ens_YYYYMMDD_HHMMSS_<6 char md5>``; we extract
# the timestamp + the trailing hash separately.  Anything that
# doesn't match falls back to filesystem mtime + a synthetic SHA.
_VERSION_PATTERN = re.compile(
    r"^ens_(?P<ts>\d{8}_\d{6})_(?P<sha>[a-fA-F0-9]+)$",
)


# ── Public entry point ────────────────────────────────────────────────


def build_governance_registry(
    registry_dir: Path,
    artifacts_dir: Path,
    active_version: str,
) -> GovernanceRegistryResponse:
    """Walk every registered ensemble version and return the registry payload.

    Parameters
    ----------
    registry_dir
        ``settings.registry_path`` — the *parent* of the ``ensemble/``
        subtree.  Mirrors the directory the eval pipeline registers
        versions into.
    artifacts_dir
        ``settings.artifacts_path`` — the *parent* of the per-version
        ``evaluation/`` subtree.  Used to discover whether a version has
        evaluation artefacts and to read its ``ensemble_metrics`` parquet.
    active_version
        The version the rest of the API is currently serving — bubbled
        up to the response so the UI can decorate that row.
    """
    registry_root = registry_dir / "ensemble"

    if not registry_root.exists():
        logger.warning("Governance: registry root missing at %s", registry_root)
        return GovernanceRegistryResponse(
            active_version=active_version, rows=[],
        )

    tag_index = _load_index(registry_root)
    version_to_tags = _invert_tag_index(tag_index)
    index_path = registry_root / _INDEX_FILENAME
    index_mtime = (
        _to_iso_utc(_safe_mtime(index_path)) if index_path.exists() else None
    )

    rows: List[GovernanceRegistryRow] = []
    for version_dir in sorted(p for p in registry_root.iterdir() if p.is_dir()):
        config_path = version_dir / _CONFIG_FILENAME
        if not config_path.exists():
            # Half-written / aborted registration — skip rather than
            # blow up the whole governance page for it.
            continue

        try:
            row = _build_row(
                version_dir=version_dir,
                config_path=config_path,
                tags=version_to_tags.get(version_dir.name, []),
                artifacts_dir=artifacts_dir,
                promoted_at=index_mtime,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception(
                "Governance: failed to build registry row for %s — skipping",
                version_dir.name,
            )
            del exc
            continue

        rows.append(row)

    rows.sort(key=lambda r: r.created_at, reverse=True)

    return GovernanceRegistryResponse(
        active_version=active_version, rows=rows,
    )


# ── Row builder ───────────────────────────────────────────────────────


def _build_row(
    *,
    version_dir: Path,
    config_path: Path,
    tags: List[str],
    artifacts_dir: Path,
    promoted_at: Optional[str],
) -> GovernanceRegistryRow:
    """Build one :class:`GovernanceRegistryRow` from one version directory."""
    version = version_dir.name

    config = json.loads(config_path.read_text())
    cluster_mapping: Dict[str, List[str]] = config.get("cluster_mapping", {}) or {}
    n_members = len(cluster_mapping)
    n_trades = sum(len(v) for v in cluster_mapping.values())
    aggregation = str(config.get("aggregation", "concat"))

    created_at = _resolve_created_at(version, version_dir)
    commit_sha = _resolve_commit_sha(version)
    status = _classify_status(tags)

    # Evaluation artefacts — best-effort.  If the parquet is missing or
    # malformed we keep the row but leave the metrics as ``None``; the
    # UI renders an em-dash.
    eval_dir = artifacts_dir / "ensemble" / version / "evaluation"
    has_evaluation = eval_dir.exists()

    mae_test, rmse_test, coverage_test = _read_test_metrics(
        eval_dir / _EVAL_METRICS_FILENAME,
    )

    return GovernanceRegistryRow(
        version=version,
        tags=tags,
        status=status,
        created_at=created_at,
        promoted_at=promoted_at if status == "production" else None,
        created_by=_DEFAULT_AUTHOR,
        commit_sha=commit_sha,
        n_members=n_members,
        n_trades=n_trades,
        aggregation=aggregation,
        has_evaluation=has_evaluation,
        mae_test=mae_test,
        rmse_test=rmse_test,
        coverage_test=coverage_test,
    )


# ── Registry index helpers ────────────────────────────────────────────


def _load_index(registry_root: Path) -> Dict[str, str]:
    """Return ``{tag: version}`` from ``index.json``, or empty dict."""
    index_path = registry_root / _INDEX_FILENAME
    if not index_path.exists():
        return {}
    try:
        return json.loads(index_path.read_text()) or {}
    except json.JSONDecodeError:
        logger.exception("Governance: malformed index.json at %s", index_path)
        return {}


def _invert_tag_index(index: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert the ``{tag: version}`` map into ``{version: [tags...]}``.

    Tags are sorted to give the response a stable order across calls.
    """
    out: Dict[str, List[str]] = {}
    for tag, version in index.items():
        out.setdefault(version, []).append(tag)
    for v in out:
        out[v].sort()
    return out


# ── Status / lifecycle classification ─────────────────────────────────


def _classify_status(tags: List[str]) -> str:
    """Bucket a tag list into one of four lifecycle states.

    Resolution order:

    1. ``production`` — the most authoritative tag wins.
    2. ``staging`` — second-rank tag.
    3. ``candidate`` — has any tag (typically ``latest``) but neither of
       the two promoted tags above.
    4. ``archived`` — no tags pointing at it (rolled past or
       superseded).
    """
    tag_set = {t.lower() for t in tags}
    if "production" in tag_set:
        return "production"
    if "staging" in tag_set:
        return "staging"
    if tag_set:
        return "candidate"
    return "archived"


# ── created_at / commit_sha resolution ────────────────────────────────


def _resolve_created_at(version: str, version_dir: Path) -> str:
    """Parse the timestamp from the version name, falling back to mtime."""
    match = _VERSION_PATTERN.match(version)
    if match:
        try:
            dt = datetime.strptime(match.group("ts"), "%Y%m%d_%H%M%S")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    return _to_iso_utc(_safe_mtime(version_dir))


def _resolve_commit_sha(version: str) -> str:
    """Return the trailing hash from a registry version, or a synthetic stub."""
    match = _VERSION_PATTERN.match(version)
    if match:
        return match.group("sha")
    # Stable but synthetic — use the directory name as the source so
    # the UI gets *something* unique instead of a placeholder string.
    return version[:8]


def _safe_mtime(path: Path) -> float:
    """Return ``path.stat().st_mtime`` with a 0-fallback so callers never raise."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _to_iso_utc(epoch_seconds: float) -> str:
    """Format a POSIX timestamp as UTC ISO-8601 string."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


# ── Test metrics extraction ───────────────────────────────────────────


def _read_test_metrics(
    metrics_path: Path,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Pull ``mae`` / ``rmse`` / ``coverage`` for the ``test`` split, if present.

    Returns ``(None, None, None)`` for any version whose evaluation
    bundle hasn't been produced (or any malformed parquet) so the
    governance row still renders.
    """
    if not metrics_path.exists():
        return None, None, None

    try:
        df = pq.read_table(metrics_path).to_pandas()
    except Exception:  # pragma: no cover — defensive
        logger.exception(
            "Governance: failed to read ensemble_metrics at %s", metrics_path,
        )
        return None, None, None

    if df.empty or "split" not in df.columns:
        return None, None, None

    test_rows = df[df["split"].astype(str).str.lower() == "test"]
    if test_rows.empty:
        return None, None, None

    row = test_rows.iloc[0]
    return (
        _maybe_float(row.get("mae")),
        _maybe_float(row.get("rmse")),
        _maybe_float(row.get("coverage")),
    )


def _maybe_float(value: Any) -> Optional[float]:
    """Coerce parquet cell to ``float`` or ``None`` (handles NaN)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
```

---

### A.3 — `src/rade_ml_pt/ensemble/api/routers/governance.py` (NEW)

```python
"""``/prism/v1/governance/*`` — cross-version registry endpoints.

The single endpoint shipped here returns one row per registered
ensemble version (the *Model Registry* table that anchors the
governance page).  Sign-offs / approvals / audit-log endpoints will
join this router in Stage 2 once the producer side exists; the route
prefix is already namespaced under ``/governance/`` so adding them is
a single ``router.get(...)`` away.

Why settings (not the reader) provide the paths
-----------------------------------------------
Every other PRISM router reads the active ensemble's bundle through
:func:`get_reader` — but governance is *cross-version* by definition.
The :class:`~src.rade_ml_pt.ensemble.api.services.reader.ArtifactReader`
holds a single :class:`~..services.paths.ArtifactPaths` keyed on the
active version, so it can't enumerate the full registry.  We pull
``registry_path`` / ``artifacts_path`` straight from settings instead
and delegate the walk to
:mod:`src.rade_ml_pt.ensemble.api.services.governance`.
"""
from __future__ import annotations

from fastapi import APIRouter

from src.rade_ml_pt.ensemble.api.config import get_settings
from src.rade_ml_pt.ensemble.api.models.governance import (
    GovernanceRegistryResponse,
)
from src.rade_ml_pt.ensemble.api.services.governance import (
    build_governance_registry,
)

router = APIRouter(prefix="/prism/v1/governance", tags=["governance"])


@router.get("/registry", response_model=GovernanceRegistryResponse)
def get_governance_registry() -> GovernanceRegistryResponse:
    """Return the full ensemble registry (one row per version)."""
    settings = get_settings()
    return build_governance_registry(
        registry_dir=settings.registry_path,
        artifacts_dir=settings.artifacts_path,
        active_version=settings.resolved_version,
    )
```

---

### A.4 — `src/rade_ml_pt/ensemble/api/app.py` (PATCH)

Two targeted edits — add the router import to the alphabetical
imports block, then include the router alongside the other artifact
routers.

**Patch 1** — find this import block:

```python
from src.rade_ml_pt.ensemble.api.routers.graph_stats import (
    router as graph_stats_router,
)
from src.rade_ml_pt.ensemble.api.routers.group_correlations import (
    router as group_correlations_router,
)
```

…and replace it with:

```python
from src.rade_ml_pt.ensemble.api.routers.governance import (
    router as governance_router,
)
from src.rade_ml_pt.ensemble.api.routers.graph_stats import (
    router as graph_stats_router,
)
from src.rade_ml_pt.ensemble.api.routers.group_correlations import (
    router as group_correlations_router,
)
```

**Patch 2** — find the tail of the router-include block in `create_app`:

```python
    app.include_router(quality_router)
    app.include_router(predictions_router)

    return app
```

…and replace it with:

```python
    app.include_router(quality_router)
    app.include_router(predictions_router)
    app.include_router(governance_router)

    return app
```

---

### A.5 — `src/rade_ml_pt/ensemble/api/client.py` (PATCH)

Two targeted edits — add the response-model import, then add a single
client method that talks to the new endpoint.

**Patch 1** — find this import block:

```python
from src.rade_ml_pt.ensemble.api.models.graph_stats import GraphStatsResponse
from src.rade_ml_pt.ensemble.api.models.group_correlations import (
    GroupCorrelationsResponse,
)
```

…and replace it with:

```python
from src.rade_ml_pt.ensemble.api.models.governance import (
    GovernanceRegistryResponse,
)
from src.rade_ml_pt.ensemble.api.models.graph_stats import GraphStatsResponse
from src.rade_ml_pt.ensemble.api.models.group_correlations import (
    GroupCorrelationsResponse,
)
```

**Patch 2** — find this section header just before the predictions
section:

```python
    # ── Predictions (binary NPZ) ──────────────────────────────────
```

…and replace it with the governance method block plus the original
section header:

```python
    # ── Governance (cross-version registry) ──────────────────────

    def governance_registry(self) -> GovernanceRegistryResponse:
        """Return one row per registered ensemble version.

        Cross-version endpoint — unlike every other method on this
        client, the response is independent of the active version
        served by the server (it covers the whole registry).  See
        :mod:`..models.governance` for the schema.
        """
        return GovernanceRegistryResponse(
            **self._get_json("/prism/v1/governance/registry")
        )

    # ── Predictions (binary NPZ) ──────────────────────────────────
```

---

### A.6 — `src/ui/apps/rade_analytics/data/backend.py` (PATCH)

Four targeted edits — model import, cache wiring, raw fetcher, and
two public accessors.

**Patch 1** — find this import block:

```python
from src.rade_ml_pt.ensemble.api.client import RadeApiClient, RadeApiError
from src.rade_ml_pt.ensemble.api.models.clusters import ClustersResponse
from src.rade_ml_pt.ensemble.api.models.meta import HealthResponse, VersionsResponse
from src.rade_ml_pt.ensemble.api.models.overview import OverviewResponse
from src.rade_ml_pt.ensemble.api.models.trade_graph import TradeGraphResponse
```

…and replace it with:

```python
from src.rade_ml_pt.ensemble.api.client import RadeApiClient, RadeApiError
from src.rade_ml_pt.ensemble.api.models.clusters import ClustersResponse
from src.rade_ml_pt.ensemble.api.models.governance import (
    GovernanceRegistryResponse,
)
from src.rade_ml_pt.ensemble.api.models.meta import HealthResponse, VersionsResponse
from src.rade_ml_pt.ensemble.api.models.overview import OverviewResponse
from src.rade_ml_pt.ensemble.api.models.trade_graph import TradeGraphResponse
```

**Patch 2** — find the tail of `_bind_cached_methods` (the
`feature_summary` line is the last existing assignment before the
governance block):

```python
        self._completeness_cached = cache.memoize(timeout=ttl)(
            self._fetch_completeness
        )
        self._feature_summary_cached = cache.memoize(timeout=ttl)(
            self._fetch_feature_summary
        )
```

…and replace it with:

```python
        self._completeness_cached = cache.memoize(timeout=ttl)(
            self._fetch_completeness
        )
        self._feature_summary_cached = cache.memoize(timeout=ttl)(
            self._fetch_feature_summary
        )
        # Governance registry — cross-version walk of the registry tree;
        # cheap on the server (one JSON read per version) but worth
        # caching at UI scope so a quick tab-flip doesn't re-issue the
        # request.  Shorter TTL (60 s) than the default because a fresh
        # ``register()`` / ``tag()`` mutation should be visible to
        # reviewers within a minute without a hard refresh.
        self._governance_registry_cached = cache.memoize(timeout=60)(
            self._fetch_governance_registry
        )
```

**Patch 3** — find this raw-fetcher block:

```python
    def _fetch_feature_summary(
        self,
        split: str,
        cluster_id: Optional[str],
    ) -> pd.DataFrame:
        resp = self._client.feature_summary(split, cluster_id=cluster_id)
        return pd.DataFrame([r.model_dump() for r in resp.rows])
```

…and replace it with:

```python
    def _fetch_feature_summary(
        self,
        split: str,
        cluster_id: Optional[str],
    ) -> pd.DataFrame:
        resp = self._client.feature_summary(split, cluster_id=cluster_id)
        return pd.DataFrame([r.model_dump() for r in resp.rows])

    def _fetch_governance_registry(self) -> GovernanceRegistryResponse:
        # Returned as the Pydantic model (not a DataFrame) because the
        # response carries two cardinalities — a scalar ``active_version``
        # used to decorate the active row and the per-version ``rows``
        # list used by the table.  Wrapping that in a ``DataFrame``
        # would lose the active-version pointer; the public
        # :meth:`governance_registry_df` flattens to a frame for
        # callbacks that only need the rows.
        return self._client.governance_registry()
```

**Patch 4** — find this public-method block (the `feature_summary_df`
accessor is the last public Quality method):

```python
    def feature_summary_df(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        return self._wrap(self._feature_summary_cached, split, cluster_id)
```

…and replace it with:

```python
    def feature_summary_df(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        return self._wrap(self._feature_summary_cached, split, cluster_id)

    # ── Governance ────────────────────────────────────────────────

    def governance_registry(
        self,
    ) -> BackendResult[GovernanceRegistryResponse]:
        """Full registry payload (active version pointer + per-version rows).

        Returned as the raw Pydantic response so the caller can decide
        whether to render the table from ``rows`` and decorate one row
        as ``active_version``, or flatten the rows to a DataFrame via
        :meth:`governance_registry_df` (most table callbacks pick the
        latter).
        """
        return self._wrap(self._governance_registry_cached)

    def governance_registry_df(self) -> BackendResult[pd.DataFrame]:
        """One-row-per-version registry as a :class:`pandas.DataFrame`.

        Stashes the ``active_version`` string on
        ``df.attrs["active_version"]`` so callbacks that need to
        highlight the active row don't have to make a second call.
        """
        res = self.governance_registry()
        if not res.ok:
            return BackendResult.failure(
                error=res.error or "", status_code=res.status_code,
            )

        rows = [r.model_dump() for r in res.data.rows]  # type: ignore[union-attr]
        df = pd.DataFrame(rows)
        df.attrs["active_version"] = res.data.active_version  # type: ignore[union-attr]
        return BackendResult.success(df)
```

---

### A.7 — `src/ui/apps/rade_analytics/layouts/governance.py` (NEW)

```python
"""Governance page layout.

Mirrors ``docs/platform_designs/rade_governance.png`` region-for-region:

* **Row 0** — invisible mount tripwire (Page Contract §3 Rule L4).
* **Row 1** — Header band: 4 KPI chips (total / production / pending /
  last-activity) on the left, status filter ``SegmentedControl`` and a
  *Promote Version* primary button on the right.
* **Row 2** — *Model Registry* AG Grid: one row per registered
  ensemble version, populated from
  :meth:`backend.governance_registry_df`.  Status / version pair render
  as a single cell with a coloured pill.
* **Row 3** — Two-column row: *Lineage Timeline* (~2/3 width, derived
  from the same registry payload + a couple of static seed events) and
  *Approvals* card (~1/3 width, static V1) housing the pending-review
  CTA, Sign-offs checklist and Policy Checks list.
* **Row 4** — *Audit Log* AG Grid (V1: hard-coded rows; the producer
  side ships in Stage 2).

Mocked vs. real
---------------
The registry table, the four header KPIs and the lineage timeline read
from the live backend.  The right-hand approvals card, the sign-off
checklist, the policy-check list and the audit-log table are static
seed data baked into this module — labelled in code where they appear
so a future commit can replace them with a real source-of-truth
without hunting.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ..components.ag_grid_table import AgGridTable
from ..components.kpi_card import KpiCard

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────
# Stable id contract — every component a callback might target lives
# here so callbacks never hardcode strings (Page Contract §3 Rule L3).
# ─────────────────────────────────────────────────────────────────────


GOVERNANCE_IDS: Dict[str, str] = {
    "root":                      "governance-root",

    # Mount tripwire — Page Contract §3 Rule L4.
    "mount_signal":              "governance-mount-signal",

    # Row 1 — Header band.
    "status_filter":             "governance-status-filter",
    "promote_btn":                "governance-promote-btn",
    "kpi_total":                 "governance-kpi-total",
    "kpi_total_value":           "governance-kpi-total-value",
    "kpi_production":            "governance-kpi-production",
    "kpi_production_value":      "governance-kpi-production-value",
    "kpi_pending":               "governance-kpi-pending",
    "kpi_pending_value":         "governance-kpi-pending-value",
    "kpi_last_activity":         "governance-kpi-last-activity",
    "kpi_last_activity_value":   "governance-kpi-last-activity-value",

    # Row 2 — Model registry grid.
    "registry_grid":             "governance-registry-grid",

    # Row 3 — Lineage timeline.
    "lineage_timeline":          "governance-lineage-timeline",

    # Row 3 — Approvals card (static V1 — id present so the Stage-2
    # render callback can swap content without a layout refactor).
    "approvals_card":            "governance-approvals-card",
    "approvals_approve_btn":     "governance-approvals-approve-btn",
    "approvals_reject_btn":      "governance-approvals-reject-btn",

    # Row 4 — Audit log grid.
    "audit_log_grid":            "governance-audit-log-grid",
}


# ─────────────────────────────────────────────────────────────────────
# Static seed data — V1 placeholders, replaced in Stage 2 when the
# producer side (audit.sqlite + sign-off workflow) ships.
# ─────────────────────────────────────────────────────────────────────


# Pending review card — single open approval.  Hard-coded version is
# not* automatically tied to the active backend version because Stage 1
# has no real workflow producer; rendering a real version here without
# real workflow state would be misleading.
_PENDING_APPROVAL = {
    "version": "v2024.04.18",
    "approvers": ["pipeline-bot", "Joe Stevens"],
}

_SIGN_OFF_CHECKLIST: List[Dict[str, str]] = [
    {"label": "Risk",       "status": "approved"},
    {"label": "Quant",      "status": "approved"},
    {"label": "Production", "status": "approved"},
]

_POLICY_CHECKS: List[Dict[str, str]] = [
    {"label": "Backtest pass",     "status": "passed"},
    {"label": "Drift within band", "status": "passed"},
    {"label": "Coverage ≥ 95%",    "status": "passed"},
    {"label": "Peer review",       "status": "pending"},
]

_AUDIT_LOG_ROWS: List[Dict[str, Any]] = [
    {
        "timestamp": "2026-04-29 18:23:33",
        "actor":     "Joe Stevens",
        "action":    "Tag",
        "target":    "production",
        "result":    "approved",
    },
    {
        "timestamp": "2026-04-29 17:08:14",
        "actor":     "pipeline-bot",
        "action":    "Eval published",
        "target":    "ens_20260429_170201",
        "result":    "approved",
    },
    {
        "timestamp": "2026-04-29 16:42:01",
        "actor":     "Joe Stevens",
        "action":    "Drift check",
        "target":    "Coverage ≥ 95%",
        "result":    "rejected",
    },
    {
        "timestamp": "2026-04-29 14:11:07",
        "actor":     "pipeline-bot",
        "action":    "Register",
        "target":    "ens_20260429_141102",
        "result":    "approved",
    },
]


# Status filter values — keep in lock-step with the
# :class:`GovernanceRegistryRow.status` enum on the backend so the
# front-end SegmentedControl filters cleanly without an extra mapping.
_STATUS_FILTER_OPTIONS = [
    {"value": "all",        "label": "All"},
    {"value": "production", "label": "Production"},
    {"value": "staging",    "label": "Staging"},
    {"value": "candidate",  "label": "Candidate"},
    {"value": "archived",   "label": "Archived"},
]
_STATUS_FILTER_DEFAULT = "all"


# ─────────────────────────────────────────────────────────────────────
# Row builders
# ─────────────────────────────────────────────────────────────────────


def _kpi_strip() -> html.Div:
    """Four KPI cards across the top of the page (left half of Row 1).

    Values are em-dashes at build time and overwritten by the bootstrap
    render callback once the registry payload arrives.
    """
    return html.Div(
        className="grid grid-cols-4 gap-4",
        children=[
            KpiCard(
                label="Total Versions",
                value="—",
                card_id=GOVERNANCE_IDS["kpi_total"],
                value_id=GOVERNANCE_IDS["kpi_total_value"],
                icon="tabler:database",
            ),
            KpiCard(
                label="In Production",
                value="—",
                card_id=GOVERNANCE_IDS["kpi_production"],
                value_id=GOVERNANCE_IDS["kpi_production_value"],
                icon="tabler:circle-check",
            ),
            KpiCard(
                label="Pending Sign-offs",
                value="—",
                card_id=GOVERNANCE_IDS["kpi_pending"],
                value_id=GOVERNANCE_IDS["kpi_pending_value"],
                icon="tabler:hourglass",
            ),
            KpiCard(
                label="Last Activity",
                value="—",
                card_id=GOVERNANCE_IDS["kpi_last_activity"],
                value_id=GOVERNANCE_IDS["kpi_last_activity_value"],
                icon="tabler:clock",
            ),
        ],
    )


def _header_actions() -> html.Div:
    """Status filter SegmentedControl + *Promote Version* button (right half).

    The filter is bound to the registry grid via a render callback that
    re-emits ``rowData``.  The promote button is wired to a no-op
    notification today; Stage 2 lifts it into the workflow producer.
    """
    return html.Div(
        className="flex items-center justify-end gap-3",
        children=[
            dmc.SegmentedControl(
                id=GOVERNANCE_IDS["status_filter"],
                data=_STATUS_FILTER_OPTIONS,
                value=_STATUS_FILTER_DEFAULT,
                size="sm",
                color="violet",
                radius="md",
            ),
            dmc.Button(
                id=GOVERNANCE_IDS["promote_btn"],
                children="Promote Version",
                color="violet",
                size="sm",
                leftSection=DashIconify(icon="tabler:rocket", width=16),
            ),
        ],
    )


def _row_header() -> html.Div:
    """Row 1 — KPI strip + filter + promote CTA in a single visual band."""
    return html.Div(
        className="flex flex-col gap-3",
        children=[
            html.Div(
                className=(
                    "flex items-center justify-between gap-4 flex-wrap"
                ),
                children=[
                    html.Div("Governance", className="rade-page-title"),
                    _header_actions(),
                ],
            ),
            _kpi_strip(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — Model Registry grid
# ─────────────────────────────────────────────────────────────────────


# Status / Result colour rules — reused by the registry + audit
# grids.  Cells get exactly one of these classes via
# ``cellClassRules`` (AG Grid evaluates the predicates server-side
# during cell render); the matching CSS lives in ``rade.css`` under
# the ``.rade-pill--*`` selectors.
_STATUS_CLASS_RULES: Dict[str, str] = {
    "rade-pill rade-pill--production": "params.value === 'production'",
    "rade-pill rade-pill--staging":    "params.value === 'staging'",
    "rade-pill rade-pill--candidate":  "params.value === 'candidate'",
    "rade-pill rade-pill--archived":   "params.value === 'archived'",
}

_AUDIT_RESULT_CLASS_RULES: Dict[str, str] = {
    "rade-pill rade-pill--approved": "params.value === 'approved'",
    "rade-pill rade-pill--rejected": "params.value === 'rejected'",
    "rade-pill rade-pill--pending":  "params.value === 'pending'",
}


_REGISTRY_COLUMN_DEFS: List[Dict[str, Any]] = [
    {
        "field": "version",
        "headerName": "Version",
        "minWidth": 220,
        "pinned": "left",
        "cellClass": "rade-grid-mono",
    },
    {
        "field": "status",
        "headerName": "Status",
        "minWidth": 130,
        "cellClassRules": _STATUS_CLASS_RULES,
        # Render-time capitalisation — keeps the wire format lowercase
        # (so the SegmentedControl filter compares cleanly) while
        # showing a human-friendly label in the cell.
        "valueFormatter": {
            "function": (
                "params.value ? "
                "params.value.charAt(0).toUpperCase() + params.value.slice(1)"
                " : '—'"
            ),
        },
    },
    {
        "field": "created_by",
        "headerName": "Created By",
        "minWidth": 140,
    },
    {
        "field": "created_at",
        "headerName": "Created",
        "minWidth": 170,
        "valueFormatter": {
            "function": (
                "params.value ? "
                "new Date(params.value).toLocaleString('en-GB', "
                "{day:'2-digit', month:'short', year:'numeric', "
                " hour:'2-digit', minute:'2-digit'}) "
                ": '—'"
            ),
        },
    },
    {
        "field": "promoted_at",
        "headerName": "Promoted",
        "minWidth": 140,
        "valueFormatter": {
            "function": (
                "params.value ? "
                "new Date(params.value).toLocaleDateString('en-GB', "
                "{day:'2-digit', month:'short', year:'numeric'}) "
                ": '—'"
            ),
        },
    },
    {
        "field": "commit_sha",
        "headerName": "Commit SHA",
        "minWidth": 110,
        "cellClass": "rade-grid-mono",
    },
    {
        "field": "mae_test",
        "headerName": "MAE (test)",
        "type": "numericColumn",
        "minWidth": 110,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toFixed(3)"
            ),
        },
    },
    {
        "field": "n_members",
        "headerName": "Clusters",
        "type": "numericColumn",
        "minWidth": 90,
    },
    {
        "field": "n_trades",
        "headerName": "Trades",
        "type": "numericColumn",
        "minWidth": 90,
        "valueFormatter": {
            "function": (
                "params.value == null ? '—' : "
                "Number(params.value).toLocaleString('en-GB')"
            ),
        },
    },
    {
        "field": "has_evaluation",
        "headerName": "Artifacts",
        "minWidth": 100,
        # Map the boolean to a glyph; AG Grid evaluates the function
        # at render time so callers can keep the wire format minimal.
        "valueFormatter": {
            "function": "params.value ? 'Available' : 'Pending'"
        },
        "cellClassRules": {
            "text-emerald-400": "params.value === true",
            "text-slate-500":   "params.value === false",
        },
    },
]


def _row_registry() -> html.Div:
    """Row 2 — full-width AG Grid for the Model Registry table."""
    return html.Div(
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div(
                "Model Registry",
                className="text-sm font-semibold text-slate-200",
            ),
            AgGridTable(
                grid_id=GOVERNANCE_IDS["registry_grid"],
                row_data=[],
                column_defs=_REGISTRY_COLUMN_DEFS,
                grid_options={
                    "pagination": True,
                    "paginationPageSize": 25,
                    "paginationPageSizeSelector": [10, 25, 50, 100],
                    "rowHeight": 40,
                    "headerHeight": 38,
                    "animateRows": False,
                    "suppressCellFocus": True,
                    "domLayout": "normal",
                },
                height=340,
                className="rade-governance-registry-grid",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 — Lineage timeline + Approvals card
# ─────────────────────────────────────────────────────────────────────


def _lineage_timeline_card() -> html.Div:
    """Left ~2/3 of Row 3 — recent lifecycle events, populated by callback.

    The body is an empty list at build time; the bootstrap render
    callback fills the timeline from the same registry payload that
    drives the table (most recent ``promoted_at`` / ``created_at``
    events get a one-line entry).  No-data fallback is rendered
    inline by the callback rather than baked here so the layout stays
    pure.
    """
    return html.Div(
        className="rade-card flex flex-col gap-3 col-span-2",
        children=[
            html.Div(
                "Lineage Timeline",
                className="text-sm font-semibold text-slate-200",
            ),
            html.Div(
                id=GOVERNANCE_IDS["lineage_timeline"],
                className="rade-feed",
                children=[],
            ),
        ],
    )


def _approvals_card() -> html.Div:
    """Right ~1/3 of Row 3 — pending approval + sign-offs + policy checks.

    Static V1 content.  The Stage-2 producer (workflow service) will
    drive this card via a callback that overwrites the
    ``approvals_card`` children — no layout change required.
    """
    avatars_row = html.Div(
        className="flex items-center gap-1",
        children=[
            DashIconify(
                icon="tabler:user-circle",
                width=22,
                className="text-slate-300",
            )
            for _ in _PENDING_APPROVAL["approvers"]
        ],
    )

    pending_block = html.Div(
        className="rade-card-compact flex flex-col gap-2",
        children=[
            html.Div(
                "Pending review:",
                className="text-xs text-slate-500 uppercase tracking-wider",
            ),
            html.Div(
                _PENDING_APPROVAL["version"],
                className="text-base font-semibold text-slate-100",
            ),
            html.Div(
                className="flex items-center justify-between gap-3 mt-1",
                children=[
                    avatars_row,
                    html.Div(
                        className="flex items-center gap-2",
                        children=[
                            dmc.Button(
                                id=GOVERNANCE_IDS["approvals_approve_btn"],
                                children="Approve",
                                color="teal",
                                size="xs",
                                variant="filled",
                            ),
                            dmc.Button(
                                id=GOVERNANCE_IDS["approvals_reject_btn"],
                                children="Reject",
                                color="red",
                                size="xs",
                                variant="light",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    sign_offs_block = html.Div(
        className="flex flex-col gap-2",
        children=[
            html.Div(
                "Sign-offs",
                className="text-sm font-semibold text-slate-200",
            ),
            *[
                html.Div(
                    className="flex items-center gap-2 text-xs text-slate-300",
                    children=[
                        DashIconify(
                            icon=(
                                "tabler:circle-check-filled"
                                if row["status"] == "approved"
                                else "tabler:circle-dashed"
                            ),
                            width=14,
                            className=(
                                "text-emerald-400"
                                if row["status"] == "approved"
                                else "text-slate-500"
                            ),
                        ),
                        html.Span(row["label"]),
                    ],
                )
                for row in _SIGN_OFF_CHECKLIST
            ],
        ],
    )

    policy_block = html.Div(
        className="flex flex-col gap-2",
        children=[
            html.Div(
                "Policy Checks",
                className="text-sm font-semibold text-slate-200",
            ),
            *[
                html.Div(
                    className="flex items-center gap-2 text-xs",
                    children=[
                        DashIconify(
                            icon=(
                                "tabler:circle-check-filled"
                                if row["status"] == "passed"
                                else "tabler:alert-triangle-filled"
                            ),
                            width=14,
                            className=(
                                "text-emerald-400"
                                if row["status"] == "passed"
                                else "text-amber-400"
                            ),
                        ),
                        html.Span(
                            row["label"],
                            className=(
                                "text-slate-300"
                                if row["status"] == "passed"
                                else "text-amber-300"
                            ),
                        ),
                        html.Span(
                            "" if row["status"] == "passed" else " · pending",
                            className="text-amber-400",
                        ),
                    ],
                )
                for row in _POLICY_CHECKS
            ],
        ],
    )

    return html.Div(
        id=GOVERNANCE_IDS["approvals_card"],
        className="rade-card flex flex-col gap-3 col-span-1",
        children=[
            html.Div(
                "Approvals",
                className="text-sm font-semibold text-slate-200",
            ),
            pending_block,
            html.Div(
                className="grid grid-cols-2 gap-4 mt-2",
                children=[sign_offs_block, policy_block],
            ),
        ],
    )


def _row_lineage_and_approvals() -> html.Div:
    return html.Div(
        className="grid grid-cols-3 gap-4 items-stretch",
        children=[
            _lineage_timeline_card(),
            _approvals_card(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — Audit log grid
# ─────────────────────────────────────────────────────────────────────


_AUDIT_LOG_COLUMN_DEFS: List[Dict[str, Any]] = [
    {
        "field": "timestamp",
        "headerName": "Timestamp",
        "minWidth": 180,
        "cellClass": "rade-grid-mono",
    },
    {"field": "actor",  "headerName": "Actor",  "minWidth": 140},
    {"field": "action", "headerName": "Action", "minWidth": 160},
    {"field": "target", "headerName": "Target", "minWidth": 200},
    {
        "field": "result",
        "headerName": "Result",
        "minWidth": 120,
        "cellClassRules": _AUDIT_RESULT_CLASS_RULES,
        "valueFormatter": {
            "function": (
                "params.value ? "
                "params.value.charAt(0).toUpperCase() + params.value.slice(1)"
                " : '—'"
            ),
        },
    },
]


def _row_audit_log() -> html.Div:
    """Row 4 — full-width AG Grid for the audit log (V1: static rows)."""
    return html.Div(
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div(
                className=(
                    "flex items-center justify-between gap-3 flex-wrap"
                ),
                children=[
                    html.Div(
                        "Audit Log (last 24h)",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Governance data sourced from ensemble registry "
                        "+ audit.sqlite",
                        className="text-xs text-slate-500",
                    ),
                ],
            ),
            AgGridTable(
                grid_id=GOVERNANCE_IDS["audit_log_grid"],
                row_data=_AUDIT_LOG_ROWS,
                column_defs=_AUDIT_LOG_COLUMN_DEFS,
                grid_options={
                    "pagination": False,
                    "rowHeight": 36,
                    "headerHeight": 38,
                    "animateRows": False,
                    "suppressCellFocus": True,
                    "domLayout": "autoHeight",
                },
                height=200,
                className="rade-governance-audit-grid",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def build_governance(*, session: Optional["Session"] = None) -> html.Div:
    """Build the full Governance page tree.

    The ``session`` kwarg is accepted for uniformity with every other
    page builder (Page Contract §2.1) but unused today — the page
    has no per-user persisted state.  Reserved so adding e.g. a
    ``governance_status_filter`` field to ``Session`` later is a
    one-line layout change.
    """
    del session  # unused today; reserved for forward-compat

    return html.Div(
        id=GOVERNANCE_IDS["root"],
        className="rade-page",
        children=[
            # Mount tripwire — Page Contract §3 Rule L4.
            dcc.Store(
                id=GOVERNANCE_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            _row_header(),
            _row_registry(),
            _row_lineage_and_approvals(),
            _row_audit_log(),
        ],
    )


__all__ = [
    "GOVERNANCE_IDS",
    "build_governance",
]
```

---

### A.8 — `src/ui/apps/rade_analytics/callbacks/governance_cb.py` (NEW)

```python
"""Governance page callbacks — live wiring for the ``/governance`` route.

Wires three rendering surfaces from a single backend call
(:meth:`RadeBackend.governance_registry`):

* **Header KPIs** — total versions / in-production / pending sign-offs
  (mocked V1) / last activity timestamp.
* **Lineage timeline** — most-recent ``promoted_at`` / ``created_at``
  events from the registry payload, plus one static seed event so the
  timeline reads non-empty even when only a single version exists.
* **Model Registry grid** — full list, filtered client-side by the
  status SegmentedControl above it.

Capture surface
---------------
The status SegmentedControl is intentionally **ephemeral** (V1) — its
selection lives only in the live component value and never makes it
into ``Session``.  That keeps this page's callback graph minimal: there
is no capture-side write, and the registry-render callback reads the
filter directly off the SegmentedControl.

If a future reviewer wants the page to remember the last filter across
navigation, add a ``governance_status_filter`` field to ``Session``,
register a small ``_sync_status_filter`` capture callback (mirror
``portfolio_cb._register_split_sync``) and seed the SegmentedControl
``value`` from session at layout build time (Page Contract §3 Rule L1).

Why mount_signal-triggered, not pathname-triggered
--------------------------------------------------
Page Contract §4 Rule C7 — when the router swaps ``app-host`` to a
new page tree, the new page's ``mount_signal`` store mounts fresh
with ``data=True``, which fires the bootstrap callback exactly once
*after* the DOM is in place.  ``Input(pathname)`` would race the
content swap and try to write into IDs that don't exist yet
(Anti-pattern A8).  Both render callbacks below trigger off
``mount_signal``; the registry-render also takes ``status_filter`` as
an Input so the table re-emits whenever the user picks a chip.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from ..layouts.governance import GOVERNANCE_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────


_GOVERNANCE_PATH = "/governance"
_PLACEHOLDER = "—"

# Pending sign-offs — V1 mock count.  Mirrors the static
# ``_PENDING_APPROVAL`` block in :mod:`..layouts.governance`.  When the
# Stage-2 workflow producer ships this becomes a backend lookup.
_MOCK_PENDING_APPROVALS = 1

# Maximum lineage timeline rows we render.  Cap is small because the
# panel is a rhythm-glance widget — denser views live on the
# Evaluation / Monitoring tabs.
_LINEAGE_MAX_ROWS = 6


# ═════════════════════════════════════════════════════════════════════
# Public surface
# ═════════════════════════════════════════════════════════════════════


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every governance callback to ``app``.

    Mirrors the Page Contract §2 capture/render split.  Capture is
    empty for V1 — every input on this page is ephemeral — so this
    function only calls :func:`_register_render`.
    """
    _register_render(app, backend)


# ─────────────────────────────────────────────────────────────────────
# Section dispatcher
# ─────────────────────────────────────────────────────────────────────


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Attach the two render callbacks (header + registry grid)."""
    _register_render_header(app, backend)
    _register_render_registry(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. Render — header KPIs + lineage timeline
# ═════════════════════════════════════════════════════════════════════


def _register_render_header(app: "Dash", backend: "RadeBackend") -> None:
    """Populate the four KPI values and the lineage timeline.

    Cheap call (cache hit after the first request) so we re-issue it
    here rather than threading the registry payload through a
    ``dcc.Store`` — keeps the callback graph linear and the diff
    against the registry-render callback obvious.
    """

    @app.callback(
        Output(GOVERNANCE_IDS["kpi_total_value"],         "children"),
        Output(GOVERNANCE_IDS["kpi_production_value"],    "children"),
        Output(GOVERNANCE_IDS["kpi_pending_value"],       "children"),
        Output(GOVERNANCE_IDS["kpi_last_activity_value"], "children"),
        Output(GOVERNANCE_IDS["lineage_timeline"],        "children"),
        Input(GOVERNANCE_IDS["mount_signal"],             "data"),
        State(SHELL_IDS["url"],                           "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _trigger: Any,
        pathname: Optional[str],
    ) -> Tuple[str, str, str, str, List[Any]]:
        # Belt-and-braces: the mount_signal Store only mounts when the
        # router swaps to the governance page tree, but a pathname
        # guard is still cheap and protects against future router
        # refactors that mount governance under a different route.
        if pathname != _GOVERNANCE_PATH:
            raise PreventUpdate

        res = backend.governance_registry()
        if not res.ok or res.data is None:
            logger.warning(
                "Governance: registry fetch failed — %s", res.error,
            )
            return (
                _PLACEHOLDER,
                _PLACEHOLDER,
                str(_MOCK_PENDING_APPROVALS),
                _PLACEHOLDER,
                _empty_lineage_message("Registry unavailable."),
            )

        rows = list(res.data.rows)
        total = len(rows)
        n_production = sum(1 for r in rows if r.status == "production")
        last_activity = _format_last_activity(rows)

        timeline = _build_lineage_timeline(rows)

        return (
            f"{total}",
            f"{n_production}",
            str(_MOCK_PENDING_APPROVALS),
            last_activity,
            timeline,
        )


# ═════════════════════════════════════════════════════════════════════
# 2. Render — Model Registry grid (filtered by status SegmentedControl)
# ═════════════════════════════════════════════════════════════════════


def _register_render_registry(app: "Dash", backend: "RadeBackend") -> None:
    """Populate the registry grid's ``rowData``, filtered by status chip.

    The status chip is *not* persisted in session (V1) — its current
    value is read straight off the SegmentedControl as a render-time
    Input.  Filtering is applied in Python before emit so AG Grid
    only ever sees the rows we want shown; this keeps the table's
    pagination + sort behaviour intuitive across filter flips.
    """

    @app.callback(
        Output(GOVERNANCE_IDS["registry_grid"], "rowData"),
        Input(GOVERNANCE_IDS["mount_signal"],   "data"),
        Input(GOVERNANCE_IDS["status_filter"],  "value"),
        State(SHELL_IDS["url"],                 "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _trigger:    Any,
        status_pick: Optional[str],
        pathname:    Optional[str],
    ) -> List[Dict[str, Any]]:
        if pathname != _GOVERNANCE_PATH:
            raise PreventUpdate

        res = backend.governance_registry_df()
        if not res.ok or res.data is None or res.data.empty:
            if not res.ok:
                logger.warning(
                    "Governance: registry_df fetch failed — %s",
                    res.error,
                )
            return []

        df = res.data
        df = _apply_status_filter(df, status_pick)

        # AG Grid expects plain dicts; ``model_dump``-derived rows
        # already JSON-friendly.  Drop any row that lost critical
        # context after the filter (defensive — shouldn't happen).
        return df.to_dict(orient="records")


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _apply_status_filter(
    df: pd.DataFrame,
    status_pick: Optional[str],
) -> pd.DataFrame:
    """Return rows matching ``status_pick``; ``"all"`` / ``None`` is identity."""
    if not status_pick or status_pick == "all":
        return df
    if "status" not in df.columns:
        return df
    return df[df["status"].astype(str) == status_pick]


def _format_last_activity(rows: List[Any]) -> str:
    """Return the most recent ``created_at`` formatted as ``DD Mon HH:MM``.

    ``rows`` is the Pydantic list straight off the response — already
    sorted newest-first by the service, so we just take the head and
    parse its ISO timestamp.  Failure to parse falls through to the
    em-dash placeholder so the KPI never renders ``Invalid Date``.
    """
    if not rows:
        return _PLACEHOLDER
    ts = getattr(rows[0], "created_at", None)
    if not ts:
        return _PLACEHOLDER
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%d %b %H:%M")


def _build_lineage_timeline(rows: List[Any]) -> List[Any]:
    """Build a small list of timeline events from the registry payload.

    Three event sources, blended into one chronologically-sorted feed:

    * **Promotion events** — every row with a non-null ``promoted_at``
      (currently the production row(s)).
    * **Registration events** — every row's ``created_at``.
    * **Evaluation events** — every row whose evaluation bundle has
      been published (``has_evaluation``); we use ``promoted_at`` as
      a proxy timestamp because we don't yet record an "evaluated_at"
      on the wire.

    Events are de-duplicated by ``(version, action)`` so a row that
    was promoted and registered doesn't generate two near-identical
    "Registered" entries.  The newest :data:`_LINEAGE_MAX_ROWS` are
    rendered.
    """
    if not rows:
        return _empty_lineage_message(
            "No registered ensemble versions yet."
        )

    events: List[Dict[str, Any]] = []
    for row in rows:
        version = row.version
        if row.promoted_at:
            events.append(
                {
                    "ts":      row.promoted_at,
                    "icon":    "tabler:rocket",
                    "kind":    "promoted",
                    "title":   f"Promoted {version} to Production",
                    "actor":   row.created_by,
                }
            )
        if row.has_evaluation and row.mae_test is not None:
            events.append(
                {
                    "ts":    row.created_at,
                    "icon":  "tabler:bolt",
                    "kind":  "evaluated",
                    "title": (
                        f"Eval completed — MAE {row.mae_test:.3f}"
                        + (
                            f"  RMSE {row.rmse_test:.3f}"
                            if row.rmse_test is not None
                            else ""
                        )
                    ),
                    "actor": "system",
                }
            )
        events.append(
            {
                "ts":     row.created_at,
                "icon":   "tabler:cube",
                "kind":   "registered",
                "title":  f"Registered {version}",
                "actor":  row.created_by,
            }
        )

    # Sort newest-first; ISO-8601 strings sort correctly lexicographically.
    events.sort(key=lambda e: str(e["ts"]), reverse=True)

    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for e in events:
        key = (e["title"], e["ts"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
        if len(deduped) >= _LINEAGE_MAX_ROWS:
            break

    return [_lineage_row(e) for e in deduped]


def _lineage_row(event: Dict[str, Any]) -> html.Div:
    """Render a single timeline event row.

    The icon's coloured background is keyed off ``event["kind"]`` —
    matches the design's coloured event marker palette.  ``ts``
    formatting matches the registry table's ``Created`` column for
    visual continuity.
    """
    icon_class = f"rade-lineage-icon rade-lineage-icon--{event['kind']}"

    body = html.Div(
        className="rade-lineage-body",
        children=[
            html.Span(event["title"]),
            html.Span(
                f"({event.get('actor', 'system')} · {_relative_time(event['ts'])})",
                className="rade-lineage-meta",
            ),
        ],
    )

    return html.Div(
        className="rade-lineage-row",
        children=[
            html.Div(
                className=icon_class,
                children=DashIconify(icon=event["icon"], width=14),
            ),
            body,
            html.A(
                "View details",
                href="#",
                className="rade-lineage-link",
            ),
        ],
    )


def _relative_time(ts: Optional[str]) -> str:
    """Return a coarse relative-time string (``2h ago`` / ``3d ago`` / ...).

    Falls back to the raw ISO string if parsing fails so the row
    still carries useful information.
    """
    if not ts:
        return _PLACEHOLDER
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta = datetime.now(tz=timezone.utc) - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 7 * 86400:
        return f"{seconds // 86400}d ago"
    return dt.strftime("%d %b %Y")


def _empty_lineage_message(message: str) -> List[Any]:
    """Single-row "no data" placeholder for the lineage timeline."""
    return [
        html.Div(
            message,
            className="text-xs text-slate-500 py-2",
        )
    ]


__all__ = ["register"]
```

---

### A.9 — `src/ui/apps/rade_analytics/router.py` (PATCH)

Two targeted edits — add the layout import, then swap the placeholder
PageSpec for the live builder.

**Patch 1** — find this import block:

```python
from .layouts.evaluation import build_evaluation
from .layouts.overview import build_overview
from .layouts.shell import SHELL_IDS, build_chrome
from .layouts.splash import build_splash
```

…and replace it with:

```python
from .layouts.evaluation import build_evaluation
from .layouts.governance import build_governance
from .layouts.overview import build_overview
from .layouts.shell import SHELL_IDS, build_chrome
from .layouts.splash import build_splash
```

**Patch 2** — find the placeholder governance entry inside `ROUTES`:

```python
    "/governance": PageSpec(
        path="/governance",
        title="Governance",
        build=_placeholder("Governance", "Phase F"),
    ),
```

…and replace it with:

```python
    "/governance": PageSpec(
        path="/governance",
        title="Governance",
        build=build_governance,
    ),
```

---

### A.10 — `src/ui/apps/rade_analytics/callbacks/__init__.py` (PATCH)

Two targeted edits — add the new module to the imports tuple, then
register it in `register_all`.

**Patch 1** — find this import block:

```python
from ..router import register_router
from . import (
    cluster_deep_dive_cb,
    evaluation_cb,
    overview_cb,
    portfolio_cb,
    splash_cb,
    trade_graph_cb,
)
```

…and replace it with:

```python
from ..router import register_router
from . import (
    cluster_deep_dive_cb,
    evaluation_cb,
    governance_cb,
    overview_cb,
    portfolio_cb,
    splash_cb,
    trade_graph_cb,
)
```

**Patch 2** — find the dispatch tail in `register_all`:

```python
    register_router(app, backend)
    splash_cb.register(app, backend)
    overview_cb.register(app, backend)
    evaluation_cb.register(app, backend)
    portfolio_cb.register(app, backend)
    trade_graph_cb.register(app, backend)
    cluster_deep_dive_cb.register(app, backend)
```

…and replace it with:

```python
    register_router(app, backend)
    splash_cb.register(app, backend)
    overview_cb.register(app, backend)
    evaluation_cb.register(app, backend)
    portfolio_cb.register(app, backend)
    trade_graph_cb.register(app, backend)
    cluster_deep_dive_cb.register(app, backend)
    governance_cb.register(app, backend)
```

---

### A.11 — `src/ui/apps/rade_analytics/assets/rade.css` (APPEND)

Append the block below to the **end** of `rade.css` (immediately
after the existing `.rade-cluster-trades-grid .ag-root-wrapper` rule).
Adds the status pills, lineage-row chrome, page title and grid
helpers used by the layout in §A.7.

```css
/* ==================================================================
 * GOVERNANCE PAGE  (Phase E.x)
 *
 * Visual anchor: ``docs/platform_designs/rade_governance.png``.
 *
 * What this section adds:
 *   .rade-page-title          — Section H1 (e.g. "Governance")
 *   .rade-grid-mono           — Monospace cell font for hashes / versions
 *   .rade-pill                — Inline pill chrome used inside AG Grid cells
 *     .rade-pill--production  — Emerald  (lifecycle: live)
 *     .rade-pill--staging     — Amber    (lifecycle: pre-prod)
 *     .rade-pill--candidate   — Violet   (lifecycle: latest, not promoted)
 *     .rade-pill--archived    — Slate    (lifecycle: rolled past)
 *     .rade-pill--approved    — Emerald  (audit result)
 *     .rade-pill--rejected    — Rose     (audit result)
 *     .rade-pill--pending     — Amber    (audit result)
 *
 * Pills sit inside AG Grid cells, so they're styled to *look* inline-
 * block-ish without overflowing the cell row.  ``cellClassRules`` in
 * ``layouts/governance.py`` wires one of the lifecycle / result
 * variants per row.
 * ================================================================== */

.rade-page-title {
  font-size: 1.125rem;
  font-weight: 700;
  color: #f1f5f9;                       /* slate-100 */
  letter-spacing: -0.01em;
}

/* Monospace cell font for AG Grid columns that show hashes / versions. */
.rade-grid-mono {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.78rem;
  letter-spacing: -0.01em;
}

/* Inline pill chrome used inside AG Grid cells.

   Setting ``display: inline-flex`` lets the pill sit alongside any
   AG-grid-injected text without inheriting the cell's full row
   height — otherwise the chip would stretch vertically and look
   like a coloured row instead of a tag.  ``align-self: center`` on
   the inner span isn't required because the cell host applies its
   own flex centering. */
.rade-pill {
  display: inline-flex;
  align-items: center;
  padding: 0.05rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  line-height: 1.5;
  border: 1px solid transparent;
}

.rade-pill--production {
  background-color: rgba(16, 185, 129, 0.16);
  color: #34d399;                       /* emerald-400 */
  border-color: rgba(16, 185, 129, 0.35);
}

.rade-pill--staging {
  background-color: rgba(245, 158, 11, 0.16);
  color: #fbbf24;                       /* amber-400 */
  border-color: rgba(245, 158, 11, 0.35);
}

.rade-pill--candidate {
  background-color: rgba(139, 92, 246, 0.16);
  color: #a78bfa;                       /* violet-400 */
  border-color: rgba(139, 92, 246, 0.35);
}

.rade-pill--archived {
  background-color: rgba(100, 116, 139, 0.16);
  color: #94a3b8;                       /* slate-400 */
  border-color: rgba(100, 116, 139, 0.35);
}

.rade-pill--approved {
  background-color: rgba(16, 185, 129, 0.16);
  color: #34d399;                       /* emerald-400 */
  border-color: rgba(16, 185, 129, 0.35);
}

.rade-pill--rejected {
  background-color: rgba(244, 63, 94, 0.16);
  color: #fb7185;                       /* rose-400 */
  border-color: rgba(244, 63, 94, 0.35);
}

.rade-pill--pending {
  background-color: rgba(245, 158, 11, 0.16);
  color: #fbbf24;                       /* amber-400 */
  border-color: rgba(245, 158, 11, 0.35);
}

/* Lineage-timeline rows — repurposes the rade-feed bullet pattern but
   bumps the icon column to 32 px so we can drop a square-ish event
   marker (e.g. "promoted", "eval completed", "registered") that
   matches the design's coloured status dots.  Markup-wise the layout
   uses ``html.Div(className="rade-lineage-row")`` for each event. */

.rade-lineage-row {
  display: grid;
  grid-template-columns: 32px 1fr auto;
  gap: 0.875rem;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(30, 41, 59, 0.6);
}

.rade-lineage-row:last-child { border-bottom: 0; }

.rade-lineage-icon {
  width: 28px;
  height: 28px;
  border-radius: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(139, 92, 246, 0.18);
  color: #a78bfa;                       /* violet-400 */
}

.rade-lineage-icon--promoted {
  background-color: rgba(16, 185, 129, 0.16);
  color: #34d399;                       /* emerald-400 */
}

.rade-lineage-icon--registered {
  background-color: rgba(139, 92, 246, 0.18);
  color: #a78bfa;                       /* violet-400 */
}

.rade-lineage-icon--evaluated {
  background-color: rgba(59, 130, 246, 0.16);
  color: #60a5fa;                       /* blue-400 */
}

.rade-lineage-body {
  font-size: 0.85rem;
  color: #cbd5e1;                       /* slate-300 */
}

.rade-lineage-meta {
  color: #64748b;                       /* slate-500 */
  font-size: 0.75rem;
  margin-left: 0.5rem;
}

.rade-lineage-link {
  font-size: 0.78rem;
  color: #818cf8;                       /* indigo-400 */
  text-decoration: none;
  white-space: nowrap;
}

.rade-lineage-link:hover { text-decoration: underline; }

/* Governance grids — match the page card and let AG Grid breathe.
   Both grids use the standard ``ag-theme-alpine-dark`` chrome so we
   only need to align the row striping and the pill cell host. */
.rade-governance-registry-grid,
.rade-governance-audit-grid {
  min-height: 0;
}

/* Centre pill cells vertically inside AG Grid (cells default to
   flex with ``align-items: center`` already, but the explicit rule
   here defends against future theme upgrades). */
.rade-governance-registry-grid .ag-cell,
.rade-governance-audit-grid .ag-cell {
  display: flex;
  align-items: center;
}
```

---

### A.12 — Verification (after pasting all eleven targets)

Run from the repo root.  The first two commands need no live registry
data; the last two do (point them at your real `PRISM_*` env vars).

1. **Static checks** — imports + lints, no live server:

   ```bash
   python -c "
   from src.rade_ml_pt.ensemble.api.app import create_app
   from src.ui.apps.rade_analytics.layouts.governance import build_governance
   from src.ui.apps.rade_analytics.callbacks import register_all
   tree = build_governance()
   assert tree.id == 'governance-root'
   print('governance imports + layout build: OK')
   "
   ```

2. **API dry-run** — synthetic 2-version registry via FastAPI's
   `TestClient` (catches Pydantic / route-prefix / response-shape regressions):

   ```bash
   python -c "
   import json, tempfile
   from pathlib import Path
   from fastapi.testclient import TestClient
   from src.rade_ml_pt.ensemble.api.config import Settings, set_settings
   from src.rade_ml_pt.ensemble.api.app import create_app
   with tempfile.TemporaryDirectory() as tmp:
       root = Path(tmp); reg = root/'registry'/'ensemble'; art = root/'artifacts'
       reg.mkdir(parents=True); art.mkdir()
       (reg/'ens_20260417_142233_abc123').mkdir()
       (reg/'ens_20260417_142233_abc123'/'ensemble_config.json').write_text(json.dumps({
           'cluster_mapping': {'cluster_0': ['t1','t2','t3']}, 'aggregation': 'concat'}))
       (reg/'index.json').write_text(json.dumps({
           'production': 'ens_20260417_142233_abc123',
           'latest':     'ens_20260417_142233_abc123'}))
       set_settings(Settings(artifacts_dir=str(root/'artifacts'),
                             registry_dir=str(root/'registry'),
                             ensemble_version='latest'))
       r = TestClient(create_app()).get('/prism/v1/governance/registry')
       assert r.status_code == 200, r.text
       body = r.json()
       assert body['rows'][0]['status'] == 'production'
       assert body['rows'][0]['n_members'] == 1
       assert body['rows'][0]['n_trades']  == 3
       print('API smoke: OK')
   "
   ```

3. **Live PRISM API** — boot the existing example launcher and call
   the endpoint:

   ```bash
   python examples/rade_ml_pt/hybrid_gnn_rnn/12_run_prism_api.py    # in one shell
   curl http://localhost:8000/prism/v1/governance/registry | jq '.rows[:2]'
   ```

4. **Live Dash UI** — launch the existing UI runner and navigate to
   `/governance`.  All four KPIs populate from the registry, the table
   row count matches `versions/active`, the SegmentedControl filters
   the table, and the Lineage Timeline shows one row per
   `(version, kind)` pair, capped at 6.
