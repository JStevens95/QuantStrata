"""Full-page layouts — one module (or sub-package) per top-level route.

Populated incrementally across Phases B–F.

Sub-modules shipped so far
--------------------------
* :mod:`.head`       — HTML ``<head>`` template consumed by ``app.py``'s
  Dash factory (``INDEX_STRING``, ``META_TAGS``, ``FONT_PRELOAD_LINKS``).
* :mod:`.shell`      — outer ``dmc.AppShell`` with sidebar, topbar and
  content slot; returns the complete ``app.layout`` tree (B.3).
* :mod:`.splash`     — pre-app entry page (hero + version picker +
  Enter CTA) exposed at ``/splash`` (C).
* :mod:`.overview`   — landing / dashboard page at ``/`` — KPI strip,
  portfolio PnL chart, cluster health heatmap, insights row, quick
  actions (D).
* :mod:`.evaluation` — global filter bar + 4 sub-tabs (Portfolio,
  Cross-Cluster, Trade-Graph, Cluster Deep-Dive) exposed under
  ``/evaluation/*`` (E.0 skeleton; real sub-tab content in E.1–E.4).
"""

from .evaluation import (
    EVALUATION_IDS,
    EVALUATION_SUBTAB_SPECS,
    active_subtab_from_path,
    build_evaluation,
    build_subtab_content,
    path_for_subtab,
)
from .head import (
    FONT_PRELOAD_LINKS,
    INDEX_STRING,
    META_TAGS,
    SVG_FAVICON_LINK,
)
from .overview import OVERVIEW_IDS, build_overview
from .shell import SHELL_IDS, build_chrome, build_root, build_shell
from .splash import SPLASH_IDS, build_splash

__all__ = [
    "EVALUATION_IDS",
    "EVALUATION_SUBTAB_SPECS",
    "FONT_PRELOAD_LINKS",
    "INDEX_STRING",
    "META_TAGS",
    "OVERVIEW_IDS",
    "SHELL_IDS",
    "SPLASH_IDS",
    "SVG_FAVICON_LINK",
    "active_subtab_from_path",
    "build_chrome",
    "build_evaluation",
    "build_overview",
    "build_root",
    "build_shell",
    "build_splash",
    "build_subtab_content",
    "path_for_subtab",
]
