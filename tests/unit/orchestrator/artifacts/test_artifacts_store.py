from __future__ import annotations

from src.orchestrator.artifacts.store import ArtifactStore


def test_artifact_store_creates_layout(tmp_path) -> None:
    store = ArtifactStore(
        workdir=tmp_path,
        run_id="run_001",
        artifacts_dirname="artifacts",
        logs_dirname="logs",
    )

    store.ensure_layout()

    # These are the public path properties provided by ArtifactStore.
    assert store.run_root.exists()
    assert store.artifacts_root.exists()
    assert store.logs_root.exists()