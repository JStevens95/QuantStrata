from __future__ import annotations

import json

from src.orchestrator.config.loader import load_run_config


def test_load_run_config_json(tmp_path) -> None:
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(
        json.dumps(
            {
                "pipeline": "dummy.pipeline",
                "dry_run": True,
                "io": {"workdir": str(tmp_path)},
                "params": {"x": 1},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_run_config(cfg_path)
    assert cfg.pipeline == "dummy.pipeline"
    assert cfg.dry_run is True
    assert cfg.params["x"] == 1