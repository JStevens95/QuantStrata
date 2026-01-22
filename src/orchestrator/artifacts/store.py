from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.orchestrator.artifacts.manifest import RunManifest


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """
    Filesystem-backed store for orchestrator run outputs.

    Layout (V1)
    -----------
    <workdir>/<run_id>/
      manifest.json
      <artifacts_dirname>/
      <logs_dirname>/

    Notes
    -----
    - This class is intentionally small and deterministic.
    - The run root is always: workdir/run_id (tests rely on this).
    """

    workdir: Path
    run_id: str
    artifacts_dirname: str = "artifacts"
    logs_dirname: str = "logs"

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    @classmethod
    def from_config(cls, *, cfg: Any, run_id: str) -> "ArtifactStore":
        """
        Build an ArtifactStore from a RunConfig-like object.

        Expected attributes (duck-typed)
        --------------------------------
        cfg.io.workdir: str
        cfg.io.artifacts_dir: str  (directory name)
        cfg.io.logs_dir: str       (directory name)
        """
        workdir = Path(str(cfg.io.workdir)).expanduser().resolve()

        # These are *dir names* (not paths) inside the run root.
        artifacts_dirname = str(getattr(cfg.io, "artifacts_dir", "artifacts"))
        logs_dirname = str(getattr(cfg.io, "logs_dir", "logs"))

        return cls(
            workdir=workdir,
            run_id=str(run_id),
            artifacts_dirname=artifacts_dirname,
            logs_dirname=logs_dirname,
        )

    # ---------------------------------------------------------------------
    # Path helpers
    # ---------------------------------------------------------------------

    @property
    def run_root(self) -> Path:
        """Root directory for this run: <workdir>/<run_id>."""
        return self.workdir / str(self.run_id)

    @property
    def artifacts_root(self) -> Path:
        """Artifacts directory for this run: <run_root>/<artifacts_dirname>."""
        return self.run_root / str(self.artifacts_dirname)

    @property
    def logs_root(self) -> Path:
        """Logs directory for this run: <run_root>/<logs_dirname>."""
        return self.run_root / str(self.logs_dirname)

    @property
    def manifest_path(self) -> Path:
        """Manifest file path: <run_root>/manifest.json."""
        return self.run_root / "manifest.json"

    # Backwards-compatible alias some code may already use
    @property
    def run_logs_path(self) -> Path:
        """Alias for logs_root (kept for compatibility during refactors)."""
        return self.logs_root

    # ---------------------------------------------------------------------
    # IO
    # ---------------------------------------------------------------------

    def ensure_layout(self) -> None:
        """
        Create the on-disk directory layout for this run.

        This is safe to call multiple times.
        """
        # Ensure run root exists.
        self.run_root.mkdir(parents=True, exist_ok=True)

        # Ensure subfolders exist.
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: RunManifest) -> Path:
        """
        Persist a RunManifest to disk atomically-ish (write tmp then replace).

        Returns
        -------
        Path
            Path to the manifest.json file.
        """
        self.ensure_layout()

        # Convert manifest to a JSON-friendly dict.
        payload: Dict[str, Any] = manifest.to_dict()  # preferred if you implemented it
        if not payload:
            # Fallback in case to_dict returns {} or isn't populated as expected.
            payload = dict(manifest.__dict__)

        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.manifest_path)

        return self.manifest_path