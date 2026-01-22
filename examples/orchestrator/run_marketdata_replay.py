"""
Example script: run marketdata.replay_static_v1

This shows how to run programmatically without requiring a CLI.

Replace dataset_path with a real dataset artifact directory.
"""

from __future__ import annotations

from src.orchestrator.config.schemas import IOConfig, RunConfig
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config


def main() -> None:
    # Construct run config in code (you can also load via load_run_config("run.yaml")).
    cfg = RunConfig(
        pipeline="marketdata.replay_static_v1",
        io=IOConfig(workdir="./.runs"),
        params={
            "dataset_path": "./path/to/marketdataset_artifact_dir",  # <-- set this
            "provider_name": "StaticProvider",
            "save_enabled": True,
            "save_name": "replayed_dataset",
            "overwrite": True,
            "static_config": {
                "strict_freq": True,
                "strict_date_coverage": True,
                "strict_scenario_coverage": True,
                "include_only_requested_ids": True,
            },
        },
    )

    # Execute pipeline.
    run_pipeline_from_config(cfg)


if __name__ == "__main__":
    main()