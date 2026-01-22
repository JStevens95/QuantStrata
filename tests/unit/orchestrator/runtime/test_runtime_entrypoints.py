from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from src.orchestrator.config.schemas import IOConfig, RunConfig
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config


@dataclass(frozen=True, slots=True)
class SetStateStep(Step):
    def run(self, ctx):
        ctx.put("hello", "world")
        return ctx


def test_run_pipeline_from_config_writes_manifest(tmp_path, monkeypatch) -> None:
    # Patch discovery to register a simple pipeline instead of importing built-ins.
    from src.orchestrator.runtime import discovery

    def _register_builtin_pipelines(registry):
        def builder(cfg):
            return Pipeline(name="dummy.pipeline", steps=[SetStateStep("set_state")])
        registry.register("dummy.pipeline", builder)

    monkeypatch.setattr(discovery, "register_builtin_pipelines", _register_builtin_pipelines)

    cfg = RunConfig(
        pipeline="dummy.pipeline",
        io=IOConfig(workdir=str(tmp_path)),
        params={},
    )

    ctx = run_pipeline_from_config(cfg, run_id="run_test_manifest")

    manifest_path = Path(tmp_path) / "run_test_manifest" / "manifest.json"
    assert manifest_path.exists()

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["pipeline"] == "dummy.pipeline"
    assert "hello" in data["outputs"]["state_keys"]

    assert ctx.get("hello") == "world"