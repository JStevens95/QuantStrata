"""
ArtifactStore.

A filesystem-backed store for:
- run folder layout
- logs folder
- artifacts folder
- manifest.json
- domain artifacts (datasets, models, reports)

We keep it small in V1 but structured for Vn extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.orchestrator.artifacts.manifest import RunManifest
from src.orchestrator.artifacts.naming import run_directory, sanitize
from src.orchestrator.artifacts.serializers import load_dataset, save_dataset
from src.orchestrator.core.errors import ArtifactError

from src.marketdata.core.dataset import MarketDataset


@dataclass(slots=True)
class ArtifactStore:
    """
    Store run outputs to the filesystem.

    Parameters
    ----------
    workdir:
        Root folder containing many runs.
    run_id:
        Unique identifier for this run (subfolder name).
    artifacts_dirname:
        Name of artifacts subfolder inside run folder.
    logs_dirname:
        Name of logs subfolder inside run folder.
    """
    workdir: Path
    run_id: str
    artifacts_dirname: str = "artifacts"
    logs_dirname: str = "logs"

    def __post_init__(self) -> None:
        # Normalize workdir to an absolute path early.
        self.workdir = Path(self.workdir).expanduser().resolve()

    @property
    def run_path(self) -> Path:
        """Full path to this run folder."""
        return run_directory(self.workdir, self.run_id)

    @property
    def artifacts_path(self) -> Path:
        """Path to the run's artifacts directory."""
        return self.run_path / sanitize(self.artifacts_dirname)

    @property
    def logs_path(self) -> Path:
        """Path to the run's logs directory."""
        return self.run_path / sanitize(self.logs_dirname)

    def ensure_layout(self) -> None:
        """
        Create run directories.

        This is idempotent (safe to call multiple times).
        """
        try:
            self.run_path.mkdir(parents=True, exist_ok=True)
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
            self.logs_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError("Failed to create run directory layout.") from exc

    # -------------------------
    # Manifest helpers
    # -------------------------

    def manifest_path(self) -> Path:
        """Canonical manifest path."""
        return self.run_path / "manifest.json"

    def save_manifest(self, manifest: RunManifest) -> Path:
        """Persist manifest to disk."""
        try:
            return manifest.save(self.manifest_path())
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError("Failed to save run manifest.") from exc

    def load_manifest(self) -> RunManifest:
        """Load manifest from disk."""
        try:
            return RunManifest.load(self.manifest_path())
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError("Failed to load run manifest.") from exc

    # -------------------------
    # Dataset helpers
    # -------------------------

    def save_market_dataset(self, dataset: MarketDataset, name: str, *, overwrite: bool = False) -> Path:
        """
        Save MarketDataset under artifacts/<name>/.

        We store datasets under a named subfolder so multiple datasets can coexist
        in a single run (useful in more complex pipelines).
        """
        target_dir = self.artifacts_path / sanitize(name)
        try:
            return save_dataset(dataset, target_dir, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(f"Failed to save MarketDataset: {name}") from exc

    def load_market_dataset(self, name: str) -> MarketDataset:
        """
        Load MarketDataset from artifacts/<name>/.
        """
        target_dir = self.artifacts_path / sanitize(name)
        try:
            return load_dataset(target_dir)
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(f"Failed to load MarketDataset: {name}") from exc