"""
Color palette for the Ensemble Analytics dark theme.

All UI modules import color tokens from here.  Never hard-code hex
values in layout or callback code.
"""

# ── Backgrounds ───────────────────────────────────────────────────
BG_PRIMARY: str = "#0d1117"
BG_CARD: str = "#161b22"
BG_HOVER: str = "#1c2333"

# ── Borders ───────────────────────────────────────────────────────
BORDER: str = "#30363d"

# ── Text ──────────────────────────────────────────────────────────
TEXT_PRIMARY: str = "#e6edf3"
TEXT_SECONDARY: str = "#8b949e"
TEXT_MUTED: str = "#484f58"

# ── Accents ───────────────────────────────────────────────────────
ACCENT_BLUE: str = "#58a6ff"
ACCENT_GREEN: str = "#3fb950"
ACCENT_RED: str = "#f85149"
ACCENT_AMBER: str = "#d29922"
ACCENT_PURPLE: str = "#bc8cff"

# ── Chart color cycle (for multi-series plots) ───────────────────
CHART_COLORS: list = [
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_AMBER,
    ACCENT_RED,
    "#79c0ff",
    "#7ee787",
    "#d2a8ff",
    "#e3b341",
    "#ffa198",
]
