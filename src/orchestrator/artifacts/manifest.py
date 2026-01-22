"""
RunManifest.

A lightweight record of:
- what pipeline ran
- when it ran
- what config was used
- what outputs were produced

Vn extensions can include:
- git hash
- package versions
- step timings
- machine metadata
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Small JSON-friendly run manifest."""
    run_id: str
    pipeline: str
    started_at_utc: str
    config: Dict[str, Any]
    outputs: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to a plain dict."""
        return dict(asdict(self))

    def save(self, path: str | Path) -> Path:
        """Persist manifest as pretty JSON."""
        p = Path(path).expanduser().resolve()
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path) -> "RunManifest":
        """Load manifest from JSON."""
        p = Path(path).expanduser().resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Manifest must contain a JSON object.")
        return RunManifest(
            run_id=str(data.get("run_id", "")),
            pipeline=str(data.get("pipeline", "")),
            started_at_utc=str(data.get("started_at_utc", "")),
            config=dict(data.get("config") or {}),
            outputs=dict(data.get("outputs") or {}),
        )