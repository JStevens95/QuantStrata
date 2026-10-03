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
