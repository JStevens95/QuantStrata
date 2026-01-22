"""
Filesystem naming helpers.

We sanitize run ids and artifact names so:
- paths are stable across OSes
- weird characters do not break folder creation
"""

from __future__ import annotations

import re
from pathlib import Path

# Replace anything not safe in file names.
_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize(value: str) -> str:
    """
    Convert an arbitrary string into a filesystem-safe slug.
    """
    cleaned = _UNSAFE_CHARS.sub("_", str(value).strip())
    return cleaned or "unnamed"


def run_directory(workdir: str | Path, run_id: str) -> Path:
    """
    Build the full run directory path under a root workdir.
    """
    root = Path(workdir).expanduser().resolve()
    return root / sanitize(run_id)