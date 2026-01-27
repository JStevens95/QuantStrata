from __future__ import annotations

import csv  # used for CSV artifact writing
import json  # used for JSON artifact writing
from dataclasses import asdict, dataclass, is_dataclass  # dataclass helpers for manifest compatibility
from pathlib import Path  # filesystem-safe path handling
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence  # type hints

from src.orchestrator.artifacts.manifest import RunManifest  # run manifest schema


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """
    Filesystem-backed store for orchestrator run outputs.

    Layout (V1)
    ----------
    <workdir>/<run_id>/
      manifest.json
      <artifacts_dirname>/
      <logs_dirname>/

    Semantics
    ---------
    - ensure_layout(): ALWAYS creates folders (tests + logging rely on this).
    - enable_save: gates write_* methods (manifest/artifacts). When False, write_* are no-ops.
    """

    workdir: Path  # base work directory (contains run folders)
    run_id: str  # unique run identifier (folder name under workdir)
    artifacts_dirname: str = "artifacts"  # subfolder name under run_root
    logs_dirname: str = "logs"  # subfolder name under run_root
    enable_save: bool = True  # default True for backwards compatibility

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    @classmethod
    def from_config(cls, *, cfg: Any, run_id: str, enable_save: Optional[bool] = None) -> "ArtifactStore":
        """
        Build an ArtifactStore from a RunConfig-like object (duck-typed).

        Expected attributes (duck-typed)
        --------------------------------
        cfg.io.workdir: str
        cfg.io.artifacts_dir: str  (directory name)
        cfg.io.logs_dir: str       (directory name)

        Parameters
        ----------
        enable_save:
            If provided, overrides config/default behaviour.
            If None, we fall back to cfg.io.enable_save if present, otherwise True.
        """
        workdir = Path(str(cfg.io.workdir)).expanduser().resolve()  # normalize to absolute path

        # These are dir *names* inside the run root.
        artifacts_dirname = str(getattr(cfg.io, "artifacts_dir", "artifacts"))  # robust default
        logs_dirname = str(getattr(cfg.io, "logs_dir", "logs"))  # robust default

        # If caller didn't pass enable_save explicitly, use cfg if available; else default True.
        resolved_enable_save = (
            bool(enable_save)
            if enable_save is not None
            else bool(getattr(cfg.io, "enable_save", True))
        )

        return cls(
            workdir=workdir,  # base directory for all runs
            run_id=str(run_id),  # folder name for this run
            artifacts_dirname=artifacts_dirname,  # artifacts folder name
            logs_dirname=logs_dirname,  # logs folder name
            enable_save=resolved_enable_save,  # write gating flag
        )

    # ---------------------------------------------------------------------
    # Path helpers
    # ---------------------------------------------------------------------

    @property
    def run_root(self) -> Path:
        """Root directory for this run: <workdir>/<run_id>."""
        return self.workdir / str(self.run_id)  # deterministic join

    @property
    def artifacts_root(self) -> Path:
        """Artifacts directory: <run_root>/<artifacts_dirname>."""
        return self.run_root / str(self.artifacts_dirname)  # deterministic join

    @property
    def logs_root(self) -> Path:
        """Logs directory: <run_root>/<logs_dirname>."""
        return self.run_root / str(self.logs_dirname)  # deterministic join

    @property
    def manifest_path(self) -> Path:
        """Manifest file path: <run_root>/manifest.json."""
        return self.run_root / "manifest.json"  # stable file name

    @property
    def run_logs_path(self) -> Path:
        """Alias for logs_root (kept for compatibility during refactors)."""
        return self.logs_root  # keep existing call sites working

    # ---------------------------------------------------------------------
    # Layout
    # ---------------------------------------------------------------------

    def ensure_layout(self) -> None:
        """
        Create the on-disk directory layout for this run.

        Notes
        -----
        - Always creates directories (tests rely on this; logger may write to logs_root).
        - Safe to call multiple times.
        """
        if self.workdir.exists() and not self.workdir.is_dir():
            # protect against accidental file path being used as a directory
            raise NotADirectoryError(f"ArtifactStore.workdir is not a directory: {self.workdir}")

        self.run_root.mkdir(parents=True, exist_ok=True)  # ensure run root exists
        self.artifacts_root.mkdir(parents=True, exist_ok=True)  # ensure artifacts dir exists
        self.logs_root.mkdir(parents=True, exist_ok=True)  # ensure logs dir exists

    def path(self, filename: str, *, in_artifacts: bool = True) -> Path:
        """
        Resolve an output path inside the run layout.
        """
        base = self.artifacts_root if in_artifacts else self.run_root  # choose base folder
        return base / str(filename)  # join base with relative filename

    # ---------------------------------------------------------------------
    # Manifest writing (backwards compatible)
    # ---------------------------------------------------------------------

    def write_manifest(self, manifest: RunManifest) -> Path:
        """
        Backwards-compatible manifest writer.

        This preserves the previous API used by runtime/entrypoints and tests.
        """
        # Convert RunManifest to a JSON-friendly dict payload.
        try:
            payload: Dict[str, Any] = manifest.to_dict()  # preferred explicit serializer
        except AttributeError:
            # Fall back to dataclasses if RunManifest is a dataclass.
            if is_dataclass(manifest):
                payload = asdict(manifest)
            else:
                raise TypeError("RunManifest must implement to_dict() or be a dataclass") from None

        return self.write_run_manifest(payload)  # delegate to the new implementation

    def write_run_manifest(self, payload: Mapping[str, Any]) -> Path:
        """
        Write a manifest payload to <run_root>/manifest.json.

        Behavior
        --------
        - If enable_save=False: no-op (returns manifest_path).
        - If enable_save=True: writes to disk (tmp then replace).
        """
        out_path = self.manifest_path  # final output path

        if not self.enable_save:
            return out_path  # no-op: return "would-be" path

        self.ensure_layout()  # ensure folders exist before writing

        tmp_path = out_path.with_suffix(".json.tmp")  # tmp file for atomic-ish replace
        tmp_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),  # stable ordering for diffs
            encoding="utf-8",  # explicit encoding
        )
        tmp_path.replace(out_path)  # replace target atomically-ish
        return out_path  # return the final manifest location

    # ---------------------------------------------------------------------
    # Artifact writers
    # ---------------------------------------------------------------------

    def write_json(self, filename: str, payload: Mapping[str, Any], *, indent: int = 2) -> Path:
        """
        Write a JSON artifact under artifacts_root/<filename>.
        """
        out_path = self.path(filename, in_artifacts=True)  # compute output path

        if not self.enable_save:
            return out_path  # no-op path return

        self.ensure_layout()  # ensure run folders exist
        out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure nested dirs exist

        with out_path.open("w", encoding="utf-8") as f:  # open file for writing
            json.dump(dict(payload), f, indent=indent, sort_keys=True)  # stable JSON

        return out_path  # return output path

    def write_csv_rows(self, filename: str, *, header: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
        """
        Write a CSV artifact under artifacts_root/<filename>.
        """
        out_path = self.path(filename, in_artifacts=True)  # compute output path

        if not self.enable_save:
            return out_path  # no-op path return

        self.ensure_layout()  # ensure run folders exist
        out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure nested dirs exist

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(header))  # fixed column order
            writer.writeheader()  # header row
            for r in rows:  # stream rows (memory-friendly)
                writer.writerow(dict(r))  # normalize and write

        return out_path  # return output path